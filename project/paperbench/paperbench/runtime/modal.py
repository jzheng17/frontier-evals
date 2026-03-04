"""Modal backend for PaperBench evaluation.

Implements ComputerRuntime and ComputerInterface using Modal Sandboxes,
providing a first-class alternative to Alcatraz/LocalConfig that runs
containers on Modal's cloud infrastructure with GPU support.

Usage via chz entrypoint:
    paperbench.solver.computer_runtime=paperbench.runtime.modal:ModalComputerRuntime
    paperbench.solver.computer_runtime.gpu_type=a10g
    paperbench.solver.computer_runtime.gpu_count=1
"""

from __future__ import annotations

import io
import tarfile
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

import structlog
from typing_extensions import override

import chz
import modal

from nanoeval.solvers.computer_tasks.code_execution_interface import (
    ComputerConfiguration,
    ComputerInterface,
    ComputerRuntime,
    ExecutionResult,
    NetworkMode,
)

logger = structlog.stdlib.get_logger(component=__name__)

SUPPORTED_NETWORK_MODES = [NetworkMode.NONE, NetworkMode.UNPROXIED]


class ModalComputerInterface(ComputerInterface):
    """ComputerInterface backed by a Modal Sandbox.

    Maps the nanoeval abstract interface directly to Modal's Sandbox API:
      send_shell_command → Sandbox.exec
      upload             → Sandbox.open (write)
      download           → Sandbox.open (read)
      disable_internet   → iptables (post-creation), or block_network at creation
      stop               → Sandbox.terminate
    """

    def __init__(self, sandbox: modal.Sandbox) -> None:
        self._sandbox = sandbox

    @override
    async def disable_internet(self) -> None:
        # If the sandbox was created with block_network=True, this is a no-op.
        # Otherwise, use iptables to drop outbound traffic (same approach as Alcatraz).
        result = await self._exec_raw(
            "iptables -I OUTPUT -j DROP 2>/dev/null && echo 'blocked' || echo 'iptables unavailable'"
        )
        output = result.unicode_output_best_effort.strip()
        if "blocked" in output:
            logger.info("Internet disabled via iptables inside Modal sandbox")
        else:
            logger.warning(
                "Could not disable internet via iptables; sandbox may still have network access",
                output=output,
            )

    @override
    async def upload(self, file: bytes, destination: str) -> None:
        # Ensure parent directory exists.
        parent = str(Path(destination).parent)
        await self._exec_raw(f"mkdir -p {parent}")

        async with await self._sandbox.open.aio(destination, "wb") as f:
            await f.write.aio(file)

    @override
    async def download(self, file: str) -> bytes:
        chunks: list[bytes] = []
        async with await self._sandbox.open.aio(file, "rb") as f:
            while True:
                chunk = await f.read.aio(8192)
                if not chunk:
                    break
                chunks.append(chunk)
        return b"".join(chunks)

    @override
    async def send_shell_command(
        self, cmd: str, *, idempotent: bool = False
    ) -> ExecutionResult:
        return await self._exec_raw(cmd)

    @override
    async def fetch_container_names(self) -> list[str]:
        return [self._sandbox.object_id]

    @override
    async def stop(self) -> None:
        try:
            await self._sandbox.terminate.aio()
        except Exception as e:
            logger.warning("Error terminating Modal sandbox", error=str(e))

    # -- internal helpers --

    async def _exec_raw(self, cmd: str) -> ExecutionResult:
        process = await self._sandbox.exec.aio("bash", "-c", cmd)
        stdout = await process.stdout.read.aio()
        stderr = await process.stderr.read.aio()
        exit_code = await process.wait.aio()
        # Combine stdout and stderr to match Alcatraz behavior where output
        # contains both streams interleaved.
        combined = stdout.encode() if isinstance(stdout, str) else stdout
        if stderr:
            stderr_bytes = stderr.encode() if isinstance(stderr, str) else stderr
            combined = combined + stderr_bytes
        return ExecutionResult(output=combined, exit_code=exit_code)


@chz.chz
class ModalComputerRuntime(ComputerRuntime):
    """ComputerRuntime backed by Modal Sandboxes.

    This is a first-class runtime backend, not a wrapper around Alcatraz.
    It implements the nanoeval ComputerRuntime interface directly using
    Modal's Sandbox API for container lifecycle management.

    Configuration is done via chz fields, which can be set from the CLI:
        paperbench.solver.computer_runtime=paperbench.runtime.modal:ModalComputerRuntime
        paperbench.solver.computer_runtime.gpu_type=a10g
        paperbench.solver.computer_runtime.gpu_count=1
        paperbench.solver.computer_runtime.dockerfile=paperbench/Dockerfile.base
    """

    # -- Image configuration --
    dockerfile: str = chz.field(
        default="paperbench/Dockerfile.base",
        doc="Path to the Dockerfile, relative to the paperbench project root.",
    )
    context_dir: str | None = chz.field(
        default=None,
        doc=(
            "Docker build context directory. Defaults to the directory containing "
            "the Dockerfile. Set to project root if the Dockerfile uses COPY from "
            "paths relative to the project."
        ),
    )
    image_tag: str = chz.field(
        default="pb-env:latest",
        doc="Image tag for identification/logging. Does not affect the built image.",
    )

    # -- GPU configuration --
    gpu_type: str | None = chz.field(
        default=None,
        doc="Modal GPU type string (e.g., 'a10g', 'a100', 'h100', 't4'). None means CPU-only.",
    )
    gpu_count: int = chz.field(
        default=0,
        doc="Number of GPUs to attach to the sandbox.",
    )

    # -- Sandbox lifecycle --
    sandbox_timeout: int = chz.field(
        default=86400,
        doc="Maximum sandbox lifetime in seconds (default: 24 hours).",
    )
    app_name: str = chz.field(
        default="__paperbench__",
        doc="Modal App name. Sandboxes are grouped under this app.",
    )

    # -- Network --
    block_network: bool = chz.field(
        default=False,
        doc="If True, create the sandbox with no outbound network access.",
    )

    # -- Environment --
    environment: dict[str, str] = chz.field(
        default_factory=dict,
        doc="Environment variables injected into the sandbox.",
    )

    # -- Docker-in-sandbox --
    enable_docker_daemon: bool = chz.field(
        default=False,
        doc=(
            "If True, start a Docker daemon inside the sandbox during runtime setup. "
            "This enables agents that need nested Docker (docker-in-sandbox). "
            "Requires the sandbox image to have Docker installed and sufficient privileges."
        ),
    )

    @override
    async def _do_runtime_setup(
        self, task: ComputerConfiguration, computer: ComputerInterface
    ) -> None:
        assert isinstance(computer, ModalComputerInterface)

        if task.network_mode not in SUPPORTED_NETWORK_MODES:
            raise ValueError(
                f"Network mode {task.network_mode} is not supported on Modal. "
                f"Supported modes: {SUPPORTED_NETWORK_MODES}"
            )

        # Disable internet if task requires it and we didn't block at creation.
        if not task.allow_internet and not self.block_network:
            logger.info("Disabling internet (task.allow_internet is False)")
            await computer.disable_internet()

        # Optionally start Docker daemon for docker-in-sandbox support.
        if self.enable_docker_daemon:
            logger.info("Starting Docker daemon inside sandbox for docker-in-sandbox support")
            await computer.send_shell_command(
                "dockerd --storage-driver=vfs &>/var/log/dockerd.log &",
                idempotent=True,
            )
            result = await computer.send_shell_command(
                "timeout 30 sh -c 'until docker info &>/dev/null; do sleep 1; done' "
                "&& echo 'docker_ready' || echo 'docker_failed'",
                idempotent=True,
            )
            output = result.unicode_output_best_effort.strip()
            if "docker_ready" in output:
                logger.info("Docker daemon started successfully inside sandbox")
            else:
                logger.warning(
                    "Docker daemon failed to start inside sandbox; "
                    "agents requiring nested Docker may not work",
                    output=output,
                )

    @override
    @asynccontextmanager
    async def _start_computer(
        self, task: ComputerConfiguration
    ) -> AsyncGenerator[ModalComputerInterface, None]:
        # Resolve Dockerfile path.
        project_root = Path(__file__).resolve().parents[2]  # paperbench project root
        dockerfile_path = project_root / self.dockerfile
        if not dockerfile_path.exists():
            raise FileNotFoundError(
                f"Dockerfile not found: {dockerfile_path}. "
                f"Ensure the path is correct relative to the paperbench project root."
            )

        context = Path(self.context_dir) if self.context_dir else dockerfile_path.parent
        if not context.is_absolute():
            context = project_root / context

        logger.info(
            "Building Modal image",
            dockerfile=str(dockerfile_path),
            context_dir=str(context),
            image_tag=self.image_tag,
        )

        # Build the image from the Dockerfile.
        image = modal.Image.from_dockerfile(
            path=dockerfile_path,
            context_dir=context,
        )

        # Resolve GPU configuration.
        gpu_config = None
        if self.gpu_count > 0 or (task.num_gpus > 0):
            gpu_count = self.gpu_count or task.num_gpus
            gpu_type = self.gpu_type or "any"
            gpu_config = f"{gpu_type}:{gpu_count}"
            logger.info("GPU configuration", gpu_config=gpu_config)

        # Merge environment variables from runtime config and task.
        merged_env = {**self.environment}
        if task.environment:
            merged_env.update(task.environment)

        # Determine network blocking.
        should_block = self.block_network or (task.network_mode == NetworkMode.NONE)

        # Create the Modal App.
        app = await modal.App.lookup.aio(
            name=self.app_name,
            create_if_missing=True,
        )

        logger.info(
            "Creating Modal sandbox",
            gpu_config=gpu_config,
            block_network=should_block,
            timeout=self.sandbox_timeout,
            env_keys=list(merged_env.keys()),
        )

        # Create the sandbox.
        sandbox = await modal.Sandbox.create.aio(
            app=app,
            image=image,
            timeout=self.sandbox_timeout,
            gpu=gpu_config,
            block_network=should_block,
            secrets=[modal.Secret.from_dict(merged_env)] if merged_env else [],
        )

        computer = ModalComputerInterface(sandbox)
        try:
            yield computer
        finally:
            logger.info("Tearing down Modal sandbox")
            await computer.stop()
