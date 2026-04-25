"""OpenHands solver for PaperBench.

Runs the OpenHands agent inside the sandbox container using the official CLI
(openhands.core.main). Unlike BasicAgent (which runs on the host and sends
commands to the sandbox), OpenHands runs entirely inside the container — making
it independent of the host's network connection after launch.

The CLI invocation matches Harbor's approach exactly: single-shot execution
with RUNTIME=local (so OpenHands uses the sandbox filesystem directly, without
Docker-in-Docker). OpenHands has its own internal agent loop that handles
multi-turn tool use, so no external re-prompting wrapper is needed.

Critical env vars aligned with Harbor (src/harbor/agents/installed/openhands.py):
  - SANDBOX_VOLUMES=${PWD}:/workspace:rw  — exposes task files to OpenHands workspace
  - SU_TO_USER=false                      — avoids user-switching in local runtime
  - USER=$(id -un)                        — sets agent identity for file operations
Without these, the agent cannot reliably find paper/instruction files in the
sandbox, leading to degraded task performance.
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
class OpenHandsSolver(BasePBSolver):
    """PaperBench solver that runs OpenHands CLI inside the sandbox container.

    Aligned with Harbor's OpenHands integration: installs openhands-ai (full
    package), invokes via `python -m openhands.core.main --task=...`, and sets
    RUNTIME=local to use the sandbox filesystem directly.
    """

    llm_model: str = chz.field(
        default="gpt-5-mini",
        doc="Model name for the OpenHands agent (e.g., gpt-5-mini, gpt-5.2)",
    )
    llm_api_key_env: str = chz.field(
        default="OPENAI_API_KEY",
        doc="Name of the host environment variable containing the API key",
    )
    llm_base_url: str | None = chz.field(
        default=None,
        doc="Optional base URL for the LLM API (for parity proxy, etc.)",
    )
    max_iterations: int = chz.field(
        default=500,
        doc="Maximum number of agent iterations (OpenHands default: 500). "
        "Aligned with Harbor's MAX_ITERATIONS env var support.",
    )
    time_limit: int = chz.field(
        default=3600,
        doc="Time limit in seconds for the agent run (default: 1 hour)",
    )
    openhands_version: str | None = chz.field(
        default=None,
        doc="Optional specific version of openhands-ai to install",
    )

    @override
    def shortname(self) -> str:
        return "openhands"

    async def _setup_computer(self, computer: ComputerInterface, task: PBTask) -> None:
        """Install OpenHands CLI in the sandbox."""
        ctx_logger = logger.bind(
            run_group_id=task.run_group_id, run_id=task.run_id, runs_dir=task.runs_dir
        )
        ctx_logger.info("Installing OpenHands in sandbox...", destinations=["run"])

        # Create a venv to avoid conflicts with task dependencies
        install_cmds = [
            "python3 -m venv /opt/openhands-venv",
            "/opt/openhands-venv/bin/pip install --upgrade pip",
        ]
        if self.openhands_version:
            install_cmds.append(
                f"/opt/openhands-venv/bin/pip install openhands-ai=={self.openhands_version}"
            )
        else:
            install_cmds.append("/opt/openhands-venv/bin/pip install openhands-ai")

        # Override openhands-ai's pinned binaryornot==0.4.4 (which has a
        # Python 3 incompatibility — `unicode()` is undefined in Py3, in
        # `binaryornot/helpers.py:106`). Observed crash: agent reads
        # paper.pdf → binaryornot detection fires → NameError → agent
        # process dies → empty submission → score 0.0 (upstream OH SSC
        # postfix run on 2026-04-25). Fix is upstream as of binaryornot
        # 0.6.0. Pin >=0.5 for safety.
        install_cmds.append(
            "/opt/openhands-venv/bin/pip install --upgrade 'binaryornot>=0.5'"
        )

        # OpenHands microagents bootstrap reads /home/.openhands_instructions
        # via FileReadAction → action_execution_server.is_binary(path) which
        # raises FileNotFoundError if the file doesn't exist. The error
        # bubbles up as a 500, RequestHTTPError kills the entire process
        # before any LLM call. Pre-creating the file as empty satisfies
        # is_binary(); the agent gets an empty microagents list and
        # proceeds normally.
        install_cmds.append(
            "mkdir -p /home/.openhands && touch /home/.openhands_instructions"
        )

        for cmd in install_cmds:
            result = await computer.send_shell_command(cmd)
            if result.exit_code != 0:
                output = result.output.decode("utf-8", errors="replace")
                raise RuntimeError(f"OpenHands install failed: {cmd}\n{output}")

        ctx_logger.info("OpenHands installed successfully", destinations=["run"])

    @override
    async def _run_agent(self, computer: ComputerInterface, task: PBTask) -> AgentOutput:
        """Run the OpenHands agent inside the sandbox via CLI."""
        ctx_logger = logger.bind(
            run_group_id=task.run_group_id, run_id=task.run_id, runs_dir=task.runs_dir
        )

        import os

        api_key = os.environ.get(self.llm_api_key_env, "")
        if not api_key:
            raise RuntimeError(f"API key env var {self.llm_api_key_env} not set on host")

        # Read instruction from the file that BasePBSolver writes
        instruction_path = "/home/instructions.txt"

        # Build env vars matching Harbor's OpenHands integration
        base_url = (
            self.llm_base_url
            or os.environ.get("LLM_BASE_URL", "")
            or os.environ.get("OPENAI_BASE_URL", "")
        )
        env_parts = [
            f"LLM_MODEL={self.llm_model}",
            f"LLM_API_KEY={api_key}",
            # Max iterations: aligned with Harbor's MAX_ITERATIONS env var
            f"MAX_ITERATIONS={self.max_iterations}",
            # Reasoning effort: aligned with Harbor's OH integration (which sets
            # LLM_REASONING_EFFORT=high by default). Without this, gpt-5/gpt-5.2
            # default to "medium", which produced systematically lower scores
            # (Harbor OH SSC mean 0.74 vs upstream 0.50 before this fix).
            "LLM_REASONING_EFFORT=high",
            # RUNTIME=local: use sandbox filesystem directly, no Docker-in-Docker
            "RUNTIME=local",
            "RUN_AS_OPENHANDS=false",
            # Workspace: mount CWD so OpenHands can find paper/instruction files
            "SANDBOX_VOLUMES=${PWD}:/workspace:rw",
            # User identity (required for local runtime file operations)
            "SU_TO_USER=false",
            "USER=$(id -un)",
            # Disable browser and prompt extensions (matching Harbor)
            "AGENT_ENABLE_PROMPT_EXTENSIONS=false",
            "AGENT_ENABLE_BROWSING=false",
            "ENABLE_BROWSER=false",
            "SANDBOX_ENABLE_AUTO_LINT=true",
            "SKIP_DEPENDENCY_CHECK=1",
            # Trajectory logging
            "SAVE_TRAJECTORY_PATH=/home/logs/openhands.trajectory.json",
            "FILE_STORE=local",
            "FILE_STORE_PATH=/home/logs/",
            "LLM_LOG_COMPLETIONS=true",
            "LLM_LOG_COMPLETIONS_FOLDER=/home/logs/completions/",
        ]
        if base_url:
            env_parts.append(f"LLM_BASE_URL={base_url}")

        # Read instruction and pass via --task flag (matching Harbor)
        read_instruction_cmd = f"cat {instruction_path}"
        result = await computer.send_shell_command(read_instruction_cmd)
        instruction = result.output.decode("utf-8", errors="replace").strip()
        # OH-only mitigation for the auto_continue trap (failure mode D).
        # When agent says "I'm done" in a message instead of calling the
        # `finish` tool, OH transitions to AWAITING_USER_INPUT and
        # `auto_continue_response` re-prompts. Agent typically declines,
        # wasting time and producing no work. Symmetric with Harbor's
        # openhands.py mitigation; framework-only fix that does NOT change
        # task design.
        instruction = instruction + (
            "\n\n"
            "IMPORTANT (OpenHands-specific): To end the task you MUST call "
            "the `finish` tool with your end_message. Do not just say 'I'm "
            "done' or 'task complete' in a regular message — that creates "
            "infinite re-prompt loops because the framework will ask "
            "'please continue' and you'll be stuck. Always invoke the "
            "finish tool to signal completion.\n"
        )
        escaped_instruction = shlex.quote(instruction)

        env_str = " ".join(env_parts)
        # Shell `timeout` wraps the agent process to ensure it's killed server-side
        # when time_limit expires. Without this, asyncio.timeout() fires but can't
        # cancel Modal's send_shell_command(), leaving the process running until
        # sandbox_timeout. Exit code 137 = killed by timeout (SIGKILL after grace).
        #
        # Redirect to file instead of piping through `tee`. With RUNTIME=local,
        # OpenHands spawns background children (Jupyter kernel, event server) that
        # inherit stdout. If those keep the pipe open after the main process exits,
        # Modal's process.stdout.read() blocks indefinitely (gRPC exec has no
        # per-read timeout), causing a spurious timeout. File redirect avoids this
        # because background children inherit the file fd, not the Modal exec pipe.
        # Matches Harbor's fix (src/harbor/agents/installed/openhands.py:1001-1018).
        run_cmd = (
            f"{env_str} timeout --kill-after=30 {self.time_limit} "
            f"/opt/openhands-venv/bin/python -m openhands.core.main"
            f" --task={escaped_instruction}"
            f" </dev/null >/home/logs/openhands.txt 2>&1"
        )

        # Ensure logs directory exists
        await computer.send_shell_command("mkdir -p /home/logs/completions")

        ctx_logger.info(
            f"Starting OpenHands agent (model={self.llm_model}, time_limit={self.time_limit}s)",
            destinations=["run"],
        )

        # Retry the OH invocation up to MAX_STARTUP_RETRIES times if the
        # agent crashes during startup (action_execution_server fails to bind
        # in time → tenacity.RetryError + httpcore.ConnectError + 0 session
        # events). Observed on 2026-04-25 for bbox + robust-clip OH runs:
        # both ran ~5min before crashing in `local_runtime._wait_until_alive`
        # without OH ever completing a single agent step. Retrying gives the
        # sandbox a fresh chance with new TCP ports / process state.
        MAX_STARTUP_RETRIES = 3
        STARTUP_FAILURE_THRESHOLD_SEC = 300  # crashes within 5min are likely startup
        start_time = time.time()
        result = None
        output = ""
        for attempt in range(1, MAX_STARTUP_RETRIES + 1):
            attempt_start = time.time()
            try:
                async with asyncio.timeout(self.time_limit):
                    result = await computer.send_shell_command(run_cmd)
                    output = result.output.decode("utf-8", errors="replace")
                    ctx_logger.info(
                        f"OpenHands agent finished (exit_code={result.exit_code})",
                        destinations=["run"],
                    )
                    # Log runner stdout for debugging (first 3000 + last 3000 chars)
                    if len(output) > 6000:
                        logged = output[:3000] + "\n...[TRUNCATED]...\n" + output[-3000:]
                    else:
                        logged = output
                    ctx_logger.info(
                        f"Runner output:\n{logged}",
                        destinations=["run"],
                    )
                    if result.exit_code != 0:
                        ctx_logger.warning(
                            f"OpenHands agent exited with non-zero code: {output[-500:]}",
                            destinations=["run"],
                        )
            except asyncio.TimeoutError:
                ctx_logger.info(
                    f"OpenHands agent timed out after {self.time_limit}s",
                    destinations=["run"],
                )
                break

            # Detect startup failure: short runtime + ConnectError signature
            attempt_runtime = time.time() - attempt_start
            oh_log_check = await computer.send_shell_command(
                "grep -c 'tenacity.RetryError\\|ConnectError' /home/logs/openhands.txt 2>/dev/null || echo 0"
            )
            # `grep -c PATTERN || echo 0` produces "0\n0" when grep finds 0
            # matches (grep exits 1, so `|| echo 0` ALSO fires). Take the
            # first line to handle that and the normal single-line case.
            raw_output = (
                oh_log_check.output.decode("utf-8", errors="replace")
                .strip()
                .split("\n")[0]
            )
            connect_err_count = int(raw_output or "0")
            if (
                attempt_runtime < STARTUP_FAILURE_THRESHOLD_SEC
                and connect_err_count > 0
                and attempt < MAX_STARTUP_RETRIES
            ):
                ctx_logger.warning(
                    f"OpenHands action_execution_server failed to start "
                    f"(attempt {attempt}/{MAX_STARTUP_RETRIES}, "
                    f"runtime={attempt_runtime:.0f}s, ConnectError detected). "
                    f"Sleeping 30s and retrying with fresh ports.",
                    destinations=["run"],
                )
                # Clear the failed log so the next attempt starts clean
                await computer.send_shell_command(
                    "mv /home/logs/openhands.txt /home/logs/openhands.txt.startup_fail.${RANDOM} 2>/dev/null || true"
                )
                await asyncio.sleep(30)
                continue
            break

        end_time = time.time()
        runtime = end_time - start_time

        # Upload the submission tarball (required by grading pipeline)
        await self._create_submission_tar(computer, task)

        ctx_logger.info(
            f"OpenHands run complete in {runtime:.0f}s",
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

        # Create tarball of submission directory (exclude .venv to avoid multi-GB
        # tarballs that corrupt during download from Modal sandboxes).
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

        # Download the tarball
        tar_bytes = await computer.download(tar_path_in_sandbox)

        # Save to the run directory
        submissions_dir = bf.join(task.run_dir, "submissions", timestamp)
        submission_path = bf.join(submissions_dir, "submission.tar.gz")
        bf.makedirs(submissions_dir)
        with bf.BlobFile(submission_path, "wb") as f:
            f.write(tar_bytes)

        ctx_logger.info(
            f"Submission tarball saved ({len(tar_bytes)} bytes): {submission_path}",
            destinations=["run"],
        )
