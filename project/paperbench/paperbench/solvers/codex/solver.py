"""Codex solver for PaperBench.

Runs the OpenAI Codex CLI agent (https://github.com/openai/codex) inside the
sandbox container in non-interactive (`codex exec`) mode. Like OpenHandsSolver,
the agent runs entirely inside the container so it does not depend on the host's
network connection after launch. Codex manages its own multi-turn agent loop.

Authentication: writes a synthetic `~/.codex/auth.json` containing the
OPENAI_API_KEY. This mirrors Harbor's Codex agent
(`src/harbor/agents/installed/codex.py`) and avoids the interactive
`codex login` flow.

CLI invocation (non-interactive):
    codex exec \
        --dangerously-bypass-approvals-and-sandbox \
        --skip-git-repo-check \
        --model gpt-5.2-codex \
        --json \
        -- <prompt>

The `--dangerously-bypass-approvals-and-sandbox` flag is necessary because the
Modal/Alcatraz sandbox is itself the isolation boundary; Codex's nested sandbox
prevents the model from running shell commands inside an already-sandboxed
container. This matches Harbor's invocation contract.
"""

import asyncio
import shlex
import time

import blobfile as bf
import structlog
from typing_extensions import override

import chz
from nanoeval.solvers.computer_tasks.code_execution_interface import ComputerInterface
from paperbench.nano.structs import AgentOutput
from paperbench.nano.task import PBTask
from paperbench.solvers.base import BasePBSolver

logger = structlog.stdlib.get_logger(component=__name__)


@chz.chz
class CodexSolver(BasePBSolver):
    """PaperBench solver that runs the Codex CLI inside the sandbox container.

    Aligned with Harbor's Codex integration: installs `@openai/codex` via npm,
    invokes via `codex exec`, and authenticates via a synthetic `auth.json`.
    """

    model: str = chz.field(
        default="gpt-5.2-codex",
        doc="Codex model to use (e.g., gpt-5.2-codex, gpt-5-codex)",
    )
    api_key_env: str = chz.field(
        default="OPENAI_API_KEY",
        doc="Name of the host environment variable containing the API key",
    )
    base_url: str | None = chz.field(
        default=None,
        doc="Optional OpenAI-compatible base URL (for parity proxy, etc.). "
        "If unset, falls back to the host's OPENAI_BASE_URL env var.",
    )
    time_limit: int = chz.field(
        default=3600,
        doc="Time limit in seconds for the agent run (default: 1 hour)",
    )
    reasoning_effort: str = chz.field(
        default="high",
        doc="Codex reasoning effort: minimal, low, medium, high",
    )
    verbosity: str = chz.field(
        default="medium",
        doc=(
            "Model text verbosity (low, medium, high). NOTE: gpt-5.2-codex "
            "ONLY accepts 'medium'; other Codex variants accept low/medium/high. "
            "Codex CLI's default is 'low', which the gpt-5.2-codex API rejects "
            "with 400 unsupported_value, so we override it here."
        ),
    )
    codex_version: str | None = chz.field(
        default=None,
        doc="Optional specific version of @openai/codex to install (e.g., '0.125.0'). "
        "If unset, installs @latest.",
    )
    node_version: str = chz.field(
        default="22",
        doc="Node.js major version to install (Codex CLI requires Node >= 20).",
    )

    @override
    def shortname(self) -> str:
        return "codex"

    async def _setup_computer(self, computer: ComputerInterface, task: PBTask) -> None:
        """Install Node.js and the Codex CLI in the sandbox."""
        ctx_logger = logger.bind(
            run_group_id=task.run_group_id, run_id=task.run_id, runs_dir=task.runs_dir
        )
        ctx_logger.info("Installing Codex CLI in sandbox...", destinations=["run"])

        # Install Node.js via NodeSource — the base image (ubuntu:24.04) ships
        # with apt's nodejs package which is too old for Codex (needs Node >= 20).
        version_spec = f"@{self.codex_version}" if self.codex_version else "@latest"
        install_script = (
            "set -euo pipefail; "
            f"curl -fsSL https://deb.nodesource.com/setup_{self.node_version}.x | bash - && "
            "apt-get install -y nodejs && "
            f"npm install -g @openai/codex{version_spec} && "
            "codex --version"
        )

        result = await computer.send_shell_command(install_script)
        if result.exit_code != 0:
            output = result.output.decode("utf-8", errors="replace")
            raise RuntimeError(f"Codex install failed (exit {result.exit_code}):\n{output[-3000:]}")

        ctx_logger.info(
            f"Codex CLI installed: {result.output.decode('utf-8', errors='replace').strip().splitlines()[-1]}",
            destinations=["run"],
        )

    @override
    async def _run_agent(self, computer: ComputerInterface, task: PBTask) -> AgentOutput:
        """Run the Codex agent inside the sandbox via `codex exec`."""
        ctx_logger = logger.bind(
            run_group_id=task.run_group_id, run_id=task.run_id, runs_dir=task.runs_dir
        )

        import os

        api_key = os.environ.get(self.api_key_env, "")
        if not api_key:
            raise RuntimeError(f"API key env var {self.api_key_env} not set on host")

        # Resolve base URL: explicit field > host env var.
        base_url = (
            self.base_url
            or os.environ.get("OPENAI_BASE_URL", "")
            or os.environ.get("LLM_BASE_URL", "")
        )

        # Read instruction from the file BasePBSolver writes (/home/instructions.txt).
        instruction_path = "/home/instructions.txt"
        result = await computer.send_shell_command(f"cat {instruction_path}")
        if result.exit_code != 0:
            raise RuntimeError(
                f"Could not read instruction at {instruction_path}: "
                f"{result.output.decode('utf-8', errors='replace')}"
            )
        instruction = result.output.decode("utf-8", errors="replace").strip()
        escaped_instruction = shlex.quote(instruction)

        # Write the synthetic auth.json that Codex CLI looks for. This mirrors
        # Harbor's pattern: cleaner than `codex login --with-api-key`, which
        # requires stdin and is harder to script in a shell pipeline.
        codex_home = "/home/agent/.codex"
        auth_setup = (
            f"mkdir -p {codex_home} /home/logs && "
            f"cat >{codex_home}/auth.json <<'EOF'\n"
            f'{{"OPENAI_API_KEY": "{api_key}"}}\n'
            f"EOF\n"
            f"chmod 600 {codex_home}/auth.json"
        )
        result = await computer.send_shell_command(auth_setup)
        if result.exit_code != 0:
            raise RuntimeError(
                f"Failed to write Codex auth.json: "
                f"{result.output.decode('utf-8', errors='replace')}"
            )

        # Build env vars for the codex process. CODEX_HOME points at our auth.json.
        env_parts = [
            f"CODEX_HOME={codex_home}",
            f"OPENAI_API_KEY={shlex.quote(api_key)}",
        ]
        if base_url:
            env_parts.append(f"OPENAI_BASE_URL={shlex.quote(base_url)}")

        env_str = " ".join(env_parts)

        # Codex CLI flags:
        #   --dangerously-bypass-approvals-and-sandbox: required since the
        #     sandbox container is the isolation boundary; Codex's nested
        #     sandbox would prevent shell-command tool calls.
        #   --skip-git-repo-check: /home is not a git repo; without this
        #     codex refuses to run.
        #   --json: emit JSONL events (consistent with Harbor's invocation,
        #     useful for downstream debugging).
        #   --cd /home: run with the workspace as the working directory so
        #     Codex sees the paper, instructions, and submission dir.
        #   -c model_reasoning_effort=...: Codex uses TOML config; this is
        #     the documented way to set reasoning effort.
        #
        # Redirect stdout/stderr to a file (not a pipe). This avoids the same
        # gRPC pipe-hang bug we hit with OpenHands: any background children
        # spawned by codex (e.g., its sandboxed exec helper) inherit the
        # stdout fd; if those keep the Modal pipe open after the main process
        # exits, `process.stdout.read()` blocks until sandbox_timeout. File
        # redirect makes background children inherit a file fd instead.
        #
        # `timeout --kill-after=30 <secs>`: server-side hard timeout. Without
        # this, asyncio.timeout() can fire on the host but cannot cancel
        # Modal's send_shell_command(), leaving the codex process orphaned.
        run_cmd = (
            f"{env_str} timeout --kill-after=30 {self.time_limit} "
            f"codex exec "
            f"--dangerously-bypass-approvals-and-sandbox "
            f"--skip-git-repo-check "
            f"--cd /home "
            f"--model {shlex.quote(self.model)} "
            f"--json "
            f"-c model_reasoning_effort={shlex.quote(self.reasoning_effort)} "
            f"-c model_verbosity={shlex.quote(self.verbosity)} "
            f"-- {escaped_instruction} "
            f"</dev/null >/home/logs/codex.jsonl 2>/home/logs/codex.stderr"
        )

        await computer.send_shell_command("mkdir -p /home/logs")

        ctx_logger.info(
            f"Starting Codex agent (model={self.model}, "
            f"reasoning_effort={self.reasoning_effort}, "
            f"time_limit={self.time_limit}s)",
            destinations=["run"],
        )

        start_time = time.time()
        try:
            async with asyncio.timeout(self.time_limit + 60):
                result = await computer.send_shell_command(run_cmd)
                ctx_logger.info(
                    f"Codex agent finished (exit_code={result.exit_code})",
                    destinations=["run"],
                )
                # Tail logs for debugging.
                tail_result = await computer.send_shell_command(
                    "echo '--- codex.stderr (tail) ---' && "
                    "tail -c 4000 /home/logs/codex.stderr 2>/dev/null || true; "
                    "echo '--- codex.jsonl (tail) ---' && "
                    "tail -c 4000 /home/logs/codex.jsonl 2>/dev/null || true"
                )
                tail = tail_result.output.decode("utf-8", errors="replace")
                ctx_logger.info(
                    f"Codex output tail:\n{tail}",
                    destinations=["run"],
                )
                if result.exit_code != 0:
                    ctx_logger.warning(
                        f"Codex agent exited with non-zero code: {result.exit_code}",
                        destinations=["run"],
                    )
        except asyncio.TimeoutError:
            ctx_logger.info(
                f"Codex agent timed out after {self.time_limit}s",
                destinations=["run"],
            )

        end_time = time.time()
        runtime = end_time - start_time

        # Best-effort: scrub the auth.json now that the agent is done so the
        # API key isn't sitting on disk in the sandbox while reproduction runs.
        try:
            await computer.send_shell_command(f"rm -f {codex_home}/auth.json")
        except Exception:
            pass

        # Upload the submission tarball (required by grading pipeline).
        await self._create_submission_tar(computer, task)

        ctx_logger.info(
            f"Codex run complete in {runtime:.0f}s",
            destinations=["group", "run"],
            _print=True,
        )

        return AgentOutput(
            run_id=task.run_id,
            time_start=start_time,
            time_end=end_time,
            error_msg=None,
            runtime_in_seconds=runtime,
            status_exists=False,
        )

    async def _create_submission_tar(self, computer: ComputerInterface, task: PBTask) -> None:
        """Create and download the submission tarball from the sandbox."""
        ctx_logger = logger.bind(
            run_group_id=task.run_group_id, run_id=task.run_id, runs_dir=task.runs_dir
        )

        from paperbench.utils import get_timestamp

        timestamp = get_timestamp()
        tar_path_in_sandbox = f"/tmp/submission_{timestamp}.tar.gz"

        # Exclude .venv to avoid multi-GB tarballs that corrupt during download
        # from Modal sandboxes (matches OpenHandsSolver).
        result = await computer.send_shell_command(
            f"tar -czf {tar_path_in_sandbox} -C /home"
            f" --exclude='submission/.venv' --exclude='submission/__pycache__'"
            f" submission"
        )
        if result.exit_code != 0:
            ctx_logger.warning(
                "Failed to create submission tarball",
                destinations=["run"],
            )
            return

        tar_bytes = await computer.download(tar_path_in_sandbox)

        submissions_dir = bf.join(task.run_dir, "submissions", timestamp)
        submission_path = bf.join(submissions_dir, "submission.tar.gz")
        bf.makedirs(submissions_dir)
        with bf.BlobFile(submission_path, "wb") as f:
            f.write(tar_bytes)

        ctx_logger.info(
            f"Submission tarball saved ({len(tar_bytes)} bytes): {submission_path}",
            destinations=["run"],
        )
