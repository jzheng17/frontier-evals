"""OpenHands solver for PaperBench.

Runs the OpenHands SDK agent inside the sandbox container. Unlike BasicAgent
(which runs on the host and sends commands to the sandbox), OpenHands runs
entirely inside the container — making it independent of the host's network
connection after launch.
"""

import asyncio
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

# Script installed into the sandbox to run the OpenHands agent
_RUNNER_SCRIPT = r'''#!/usr/bin/env python3
"""Runner script for OpenHands SDK inside the PaperBench sandbox."""

import json
import os
import sys
from pathlib import Path

def main():
    instruction_path = sys.argv[1] if len(sys.argv) > 1 else "/home/instructions.txt"
    instruction = Path(instruction_path).read_text().strip()

    model = os.environ.get("LLM_MODEL", "gpt-5-mini")
    api_key = os.environ.get("LLM_API_KEY", "")
    base_url = os.environ.get("LLM_BASE_URL", "")

    if not api_key:
        print("ERROR: LLM_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    from openhands.sdk import LLM, Agent, Conversation, Tool
    from openhands.tools.terminal import TerminalTool
    from openhands.tools.file_editor import FileEditorTool

    llm_kwargs = {"model": model, "api_key": api_key}
    if base_url:
        llm_kwargs["base_url"] = base_url
    llm = LLM(**llm_kwargs)

    tools = [
        Tool(name=TerminalTool.name),
        Tool(name=FileEditorTool.name),
    ]

    agent = Agent(llm=llm, tools=tools)
    workspace = "/home"
    conversation = Conversation(agent=agent, workspace=workspace)

    print(f"Starting OpenHands agent with model={model}")
    print(f"Instruction: {instruction[:200]}...")

    conversation.send_message(instruction)
    conversation.run()

    # Save metrics
    token_usage = llm.metrics.accumulated_token_usage
    metrics = {
        "prompt_tokens": token_usage.prompt_tokens if token_usage else 0,
        "completion_tokens": token_usage.completion_tokens if token_usage else 0,
        "cost_usd": llm.metrics.accumulated_cost,
    }
    metrics_path = Path("/home/logs/openhands_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Agent completed. Cost: ${metrics['cost_usd']:.4f}")

if __name__ == "__main__":
    main()
'''


@chz.chz
class OpenHandsSolver(BasePBSolver):
    """PaperBench solver that runs OpenHands SDK inside the sandbox container."""

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
    time_limit: int = chz.field(
        default=3600,
        doc="Time limit in seconds for the agent run (default: 1 hour)",
    )
    openhands_version: str | None = chz.field(
        default=None,
        doc="Optional specific version of openhands-sdk to install",
    )

    @override
    def shortname(self) -> str:
        return "openhands"

    async def _setup_computer(self, computer: ComputerInterface, task: PBTask) -> None:
        """Install OpenHands SDK in the sandbox."""
        ctx_logger = logger.bind(
            run_group_id=task.run_group_id, run_id=task.run_id, runs_dir=task.runs_dir
        )
        ctx_logger.info("Installing OpenHands SDK in sandbox...", destinations=["run"])

        # Create a venv to avoid conflicts with task dependencies
        install_cmds = [
            "python3 -m venv /opt/openhands-venv",
            "/opt/openhands-venv/bin/pip install --upgrade pip",
        ]
        if self.openhands_version:
            install_cmds.append(
                f"/opt/openhands-venv/bin/pip install openhands-sdk=={self.openhands_version} "
                f"openhands-tools=={self.openhands_version}"
            )
        else:
            install_cmds.append(
                "/opt/openhands-venv/bin/pip install openhands-sdk openhands-tools"
            )

        for cmd in install_cmds:
            result = await computer.send_shell_command(cmd)
            if result.exit_code != 0:
                output = result.output.decode("utf-8", errors="replace")
                raise RuntimeError(f"OpenHands install failed: {cmd}\n{output}")

        # Upload the runner script
        await computer.upload(
            _RUNNER_SCRIPT.encode("utf-8"),
            "/opt/openhands-venv/run_agent.py",
        )

        ctx_logger.info("OpenHands SDK installed successfully", destinations=["run"])

    @override
    async def _run_agent(self, computer: ComputerInterface, task: PBTask) -> AgentOutput:
        """Run the OpenHands agent inside the sandbox."""
        ctx_logger = logger.bind(
            run_group_id=task.run_group_id, run_id=task.run_id, runs_dir=task.runs_dir
        )

        import os

        api_key = os.environ.get(self.llm_api_key_env, "")
        if not api_key:
            raise RuntimeError(
                f"API key env var {self.llm_api_key_env} not set on host"
            )

        # Build env vars for the agent process
        env_parts = [
            f"LLM_MODEL={self.llm_model}",
            f"LLM_API_KEY={api_key}",
        ]
        if self.llm_base_url:
            env_parts.append(f"LLM_BASE_URL={self.llm_base_url}")

        env_str = " ".join(env_parts)
        run_cmd = (
            f"{env_str} /opt/openhands-venv/bin/python "
            f"/opt/openhands-venv/run_agent.py /home/instructions.txt"
        )

        ctx_logger.info(
            f"Starting OpenHands agent (model={self.llm_model}, "
            f"time_limit={self.time_limit}s)",
            destinations=["run"],
        )

        start_time = time.time()
        try:
            async with asyncio.timeout(self.time_limit):
                result = await computer.send_shell_command(run_cmd)
                output = result.output.decode("utf-8", errors="replace")
                ctx_logger.info(
                    f"OpenHands agent finished (exit_code={result.exit_code})",
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

    async def _create_submission_tar(
        self, computer: ComputerInterface, task: PBTask
    ) -> None:
        """Create and download the submission tarball from the sandbox."""
        ctx_logger = logger.bind(
            run_group_id=task.run_group_id, run_id=task.run_id, runs_dir=task.runs_dir
        )

        from paperbench.utils import get_timestamp

        timestamp = get_timestamp()
        tar_path_in_sandbox = f"/tmp/submission_{timestamp}.tar.gz"

        # Create tarball of submission directory
        result = await computer.send_shell_command(
            f"tar -czf {tar_path_in_sandbox} -C /home submission"
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
        bf.makedirs(submissions_dir, exist_ok=True)
        with bf.BlobFile(submission_path, "wb") as f:
            f.write(tar_bytes)

        ctx_logger.info(
            f"Submission tarball saved ({len(tar_bytes)} bytes): {submission_path}",
            destinations=["run"],
        )
