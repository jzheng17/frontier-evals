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

import asyncio
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

import structlog
from typing_extensions import override

import chz
import modal

modal.enable_output()

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
    async def upload(self, file: bytes, destination: str, timeout: float = 600, max_retries: int = 3) -> None:
        """Upload bytes to the sandbox with timeout + retry.

        Mirrors download()'s timeout/retry pattern (and Harbor PR #1548's
        modal upload fix). Without this, a single hung Modal `open.aio()`
        or `f.write.aio()` (TCP/control-plane glitch) wedges the trial
        forever — observed empirically on the SSNE/t1 upstream Tier 2
        trial 2026-04-29.
        """
        # Ensure parent directory exists.
        parent = str(Path(destination).parent)
        await self._exec_raw(f"mkdir -p {parent}")

        last_exc: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                async def _do_write() -> None:
                    async with await self._sandbox.open.aio(destination, "wb") as f:
                        await f.write.aio(file)
                await asyncio.wait_for(_do_write(), timeout=timeout)
                return
            except asyncio.TimeoutError as e:
                last_exc = e
                logger.warning(
                    "Upload timed out",
                    destination=destination, timeout=timeout,
                    attempt=attempt, max_retries=max_retries,
                )
            except Exception as e:
                last_exc = e
                logger.warning(
                    "Upload failed",
                    destination=destination, error=str(e),
                    attempt=attempt, max_retries=max_retries,
                )

            if attempt < max_retries:
                # Exponential backoff: 2, 4, 8 ... capped at 30s
                wait_s = min(2 ** attempt, 30)
                await asyncio.sleep(wait_s)

        raise last_exc if last_exc is not None else RuntimeError(
            f"Upload to {destination} failed after {max_retries} attempts"
        )

    @override
    async def download(self, file: str, timeout: float = 600, max_retries: int = 3) -> bytes:
        """Download a file from the sandbox with retries and timeout.

        Tries Sandbox.filesystem API first (preferred), then falls back to
        chunked FileIO.read. Each method is retried up to max_retries times
        with exponential backoff on timeout/transient errors.
        """
        has_filesystem_api = hasattr(self._sandbox, "filesystem")
        last_exc: Exception | None = None

        for attempt in range(1, max_retries + 1):
            # Try the newer filesystem API first
            if has_filesystem_api:
                try:
                    data = await asyncio.wait_for(
                        self._sandbox.filesystem.read_bytes.aio(file),
                        timeout=timeout,
                    )
                    return data
                except asyncio.TimeoutError as e:
                    last_exc = e
                    logger.warning(
                        "Download timed out via filesystem API",
                        file=file, timeout=timeout,
                        attempt=attempt, max_retries=max_retries,
                    )
                except Exception as e:
                    last_exc = e
                    logger.warning(
                        "Download failed via filesystem API",
                        file=file, error=str(e),
                        attempt=attempt, max_retries=max_retries,
                    )

            # Fallback: chunked read
            try:
                chunks: list[bytes] = []

                async def _chunked_read() -> bytes:
                    async with await self._sandbox.open.aio(file, "rb") as f:
                        while True:
                            chunk = await f.read.aio(1024 * 1024)
                            if not chunk:
                                break
                            chunks.append(chunk)
                    return b"".join(chunks)

                return await asyncio.wait_for(_chunked_read(), timeout=timeout)
            except asyncio.TimeoutError as e:
                last_exc = e
                logger.warning(
                    "Download timed out via chunked read",
                    file=file, timeout=timeout,
                    attempt=attempt, max_retries=max_retries,
                )
            except Exception as e:
                last_exc = e
                logger.warning(
                    "Download failed via chunked read",
                    file=file, error=str(e),
                    attempt=attempt, max_retries=max_retries,
                )

            if attempt < max_retries:
                backoff = 2 ** attempt
                logger.info(f"Retrying download in {backoff}s...", file=file)
                await asyncio.sleep(backoff)

        raise last_exc or RuntimeError(f"Download failed after {max_retries} retries: {file}")

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

    async def _exec_raw(
        self, cmd: str, timeout: int | None = None
    ) -> ExecutionResult:
        process = await self._sandbox.exec.aio(
            "bash", "-c", cmd, timeout=timeout
        )
        # Wrap stdout/stderr/wait reads with explicit timeouts so a hung
        # Modal pipe (sandbox dying mid-call, control-plane glitch, etc.)
        # raises asyncio.TimeoutError instead of stalling the trial forever.
        # Mirrors Harbor PR #1548's modal exec fix; observed empirically
        # on SSNE/t1 upstream Tier 2 (2026-04-29) where a post-trial
        # in-sandbox tar command hung 3.5h until sandbox_timeout fired.
        # 600s per read is generous; healthy reads complete in milliseconds.
        read_timeout = 600.0
        try:
            stdout_raw = await asyncio.wait_for(
                process.stdout.read.aio(), timeout=read_timeout
            )
        except asyncio.TimeoutError:
            stdout_raw = b""
        try:
            stderr_raw = await asyncio.wait_for(
                process.stderr.read.aio(), timeout=read_timeout
            )
        except asyncio.TimeoutError:
            stderr_raw = b""
        try:
            exit_code = await asyncio.wait_for(
                process.wait.aio(), timeout=read_timeout
            )
        except asyncio.TimeoutError:
            exit_code = -1
        # Combine stdout and stderr to match Alcatraz behavior where output
        # contains both streams interleaved. Decode bytes with errors="replace"
        # so malformed UTF-8 (binary blobs, truncated multibyte chars) doesn't
        # crash downstream Pydantic / log handling.
        if isinstance(stdout_raw, bytes):
            combined = stdout_raw
        else:
            combined = stdout_raw.encode("utf-8", errors="replace")
        if stderr_raw:
            stderr_bytes = (
                stderr_raw if isinstance(stderr_raw, bytes)
                else stderr_raw.encode("utf-8", errors="replace")
            )
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
    """

    # -- Image configuration --
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
        doc="Modal App name prefix. A unique suffix is appended per instance to avoid sandbox interference between concurrent runs.",
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

    def _build_image(self) -> modal.Image:
        """Build the PaperBench sandbox image using Modal's programmatic API.

        Equivalent to Dockerfile.base but avoids from_dockerfile() which has
        opaque build failures. Each step mirrors a RUN/COPY in the Dockerfile.
        """
        project_root = Path(__file__).resolve().parents[2]

        # Start from Ubuntu 24.04
        image = modal.Image.from_registry("ubuntu:24.04")

        # Directory structure matching Dockerfile.base ENV/mkdir
        image = image.run_commands(
            "mkdir -p /home/logs /home/agent /home/submission /home/paper /home/.vscode "
            "/submission /output"
        )

        # COPY local files (launch.json, pre-commit, apply_patch.py).
        # copy=True embeds files in the image layer so run_commands can use them.
        image = image.add_local_file(
            str(project_root / "paperbench" / "solvers" / "launch.json"),
            "/home/.vscode/launch.json",
            copy=True,
        )
        image = image.add_local_file(
            str(project_root / "paperbench" / "solvers" / "apply_patch.py"),
            "/home/agent/apply_patch.py",
            copy=True,
        )
        image = image.add_local_file(
            str(project_root / "paperbench" / "solvers" / "pre-commit"),
            "/home/submission/.git/hooks/pre-commit",
            copy=True,
        )

        # System packages (matches apt-get install block in Dockerfile.base)
        image = image.run_commands(
            "export DEBIAN_FRONTEND=noninteractive && "
            "apt-get update && apt-get install -y "
            "curl wget git vim nano unzip zip p7zip-full "
            "python3 python3-pip python3-venv python3-dev python-is-python3 "
            "build-essential openssh-server tmux gettext sudo ffmpeg libsm6 libxext6 "
            "&& rm -rf /var/lib/apt/lists/*"
        )
        image = image.run_commands(
            "export DEBIAN_FRONTEND=noninteractive && apt update && apt install -y jupyter"
        )

        # Docker daemon (for docker-in-sandbox support)
        image = image.run_commands(
            "curl -fsSL https://get.docker.com -o /tmp/get-docker.sh && "
            "chmod 700 /tmp/get-docker.sh && /tmp/get-docker.sh"
        )

        # Miniconda (always x86_64 on Modal)
        image = image.run_commands(
            'wget "https://repo.anaconda.com/miniconda/Miniconda3-py313_25.5.1-0-Linux-x86_64.sh" '
            "-O /tmp/miniconda.sh && "
            "bash /tmp/miniconda.sh -b -p /opt/conda && "
            "rm /tmp/miniconda.sh && "
            "/opt/conda/bin/conda init"
        )

        # Conda env creation
        image = image.run_commands(
            "/opt/conda/bin/conda tos accept --override-channels "
            "--channel https://repo.anaconda.com/pkgs/main && "
            "/opt/conda/bin/conda tos accept --override-channels "
            "--channel https://repo.anaconda.com/pkgs/r && "
            "/opt/conda/bin/conda create -n agent python=3.12 -y"
        )

        # Git setup for submission directory
        image = image.run_commands(
            "cd /home/submission && git init && mkdir -p .git/hooks && "
            "chmod +x .git/hooks/pre-commit && "
            'git config --global user.email "agent@example.com" && '
            'git config --global user.name "agent"'
        )

        # apply_patch helper
        image = image.run_commands(
            "echo '#!/bin/bash' > /bin/apply_patch && "
            """echo 'python /home/agent/apply_patch.py "$@"' >> /bin/apply_patch && """
            "chmod +x /bin/apply_patch"
        )

        # Environment variables matching Dockerfile.base
        image = image.env({
            "WORKSPACE_BASE": "/home",
            "SUBMISSION_DIR": "/home/submission",
            "LOGS_DIR": "/home/logs",
            "AGENT_DIR": "/home/agent",
            "CONDA_ENV_NAME": "agent",
            "REQUIREMENTS": "/home/agent/requirements.txt",
            "PYTHON_VERSION": "3.12",
            "PATH": "/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        })

        image = image.workdir("/home")

        return image

    def _build_reproducer_image(self) -> modal.Image:
        """Build the PaperBench reproducer image using Modal's programmatic API.

        Extends the base pb-env image with reproducer-specific layers:
        Python 3.11/3.12 via deadsnakes, build tools, and pip bootstrapping.
        Mirrors reproducer.Dockerfile.
        """
        image = self._build_image()

        # Build tools and ML dependency packages (reproducer.Dockerfile lines 9-16)
        image = image.run_commands(
            "export DEBIAN_FRONTEND=noninteractive && "
            "apt-get update && apt-get install -y "
            "software-properties-common wget curl unzip sudo "
            "build-essential git cmake "
            "libatlas-base-dev libblas-dev liblapack-dev libopenblas-dev "
            "gfortran libsm6 libxext6 libxrender-dev "
            "&& rm -rf /var/lib/apt/lists/*"
        )

        # Python 3.11 and 3.12 via deadsnakes PPA (reproducer.Dockerfile lines 19-25)
        image = image.run_commands(
            "export DEBIAN_FRONTEND=noninteractive && "
            "add-apt-repository ppa:deadsnakes/ppa && "
            "apt-get update && apt-get install -y "
            "python3.11 python3.11-venv python3.11-dev "
            "python3.12 python3.12-venv python3.12-dev "
            "python3-pip "
            "&& rm -rf /var/lib/apt/lists/*"
        )

        # update-alternatives for python version switching (reproducer.Dockerfile lines 28-29)
        image = image.run_commands(
            "update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && "
            "update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 2"
        )

        # Ensure python/pip symlinks (reproducer.Dockerfile lines 33-34)
        image = image.run_commands(
            "ln -sf /usr/bin/python3 /usr/bin/python && "
            "ln -sf /usr/bin/pip3 /usr/bin/pip || true"
        )

        # Hard guarantee: python + pip CLIs exist (reproducer.Dockerfile lines 42-57)
        image = image.run_commands(
            "export DEBIAN_FRONTEND=noninteractive && "
            "apt-get update && apt-get install -y --no-install-recommends "
            "python3-pip python-is-python3 "
            "&& rm -rf /var/lib/apt/lists/*"
        )
        image = image.run_commands(
            "python3 -m pip --version || ("
            "  python3 -m ensurepip --upgrade || true;"
            "  python3 -m pip --version || ("
            "    curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py &&"
            "    python3 /tmp/get-pip.py"
            "  )"
            ")"
        )
        image = image.run_commands(
            "command -v pip || ln -sf $(command -v pip3) /usr/local/bin/pip || true"
        )

        return image

    @override
    @asynccontextmanager
    async def _start_computer(
        self, task: ComputerConfiguration
    ) -> AsyncGenerator[ModalComputerInterface, None]:
        # Dispatch image builder based on task.docker_image.
        docker_image = task.docker_image or ""
        if "reproducer" in docker_image:
            logger.info("Building Modal reproducer image", image_tag=docker_image)
            image = self._build_reproducer_image()
        else:
            logger.info("Building Modal image", image_tag=self.image_tag)
            image = self._build_image()

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

        # Create the Modal App with a unique name per sandbox to prevent
        # concurrent runs from interfering with each other's sandboxes.
        unique_app_name = f"{self.app_name}-{uuid.uuid4().hex[:8]}"
        app = await modal.App.lookup.aio(
            name=unique_app_name,
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
