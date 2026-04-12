# Modal Runtime & Image Architecture

## Overview

This fork adds a Modal-based `ComputerRuntime` (`ModalComputerRuntime`) as a first-class
alternative to Alcatraz for running PaperBench experiments. Modal provides on-demand cloud
GPUs and sandboxed containers without requiring local Docker or GPU hardware.

## Two-Layer Image Architecture

PaperBench containers are built from two layers that correspond directly to the existing
Dockerfile structure in the upstream repo:

### Layer 1: Pre-built base images (GHCR)

These images are built from the **exact same Dockerfiles** already in this repo:

| Image | Built from | Contents |
|-------|-----------|----------|
| `ghcr.io/jzheng17/paperbench-env:latest` | `paperbench/Dockerfile.base` | Ubuntu 24.04, conda, Docker daemon, Python 3.12, git hooks, system packages |
| `ghcr.io/jzheng17/paperbench-reproducer:latest` | `paperbench/reproducer.Dockerfile` | Extends pb-env with Python 3.11+3.12 (deadsnakes), BLAS libraries, pip bootstrapping |

The pre-built images are **byte-identical** to what `docker build -f Dockerfile.base .`
produces locally. No packages are added, removed, or modified.

### Layer 2: Task-specific resources (programmatic, per-run)

Task-specific files are added at runtime via Modal's programmatic image API:

```python
image = modal.Image.from_registry("ghcr.io/jzheng17/paperbench-env:latest")
image = image.add_local_file(paper_path, "/home/paper/paper.pdf", copy=True)
image = image.add_local_file(rubric_path, "/home/paper/rubric.json", copy=True)
# ... agent-specific files
```

This is the programmatic equivalent of `FROM pb-env:latest` followed by `COPY` in a
Dockerfile. No behavioral changes are introduced.

## Why pre-built instead of fully programmatic?

The previous approach built `pb-env` from scratch on every Modal run using
`modal.Image.from_registry("ubuntu:24.04")` followed by ~90 lines of `run_commands()`.
This had three problems:

1. **Build failures**: Modal's remote builder cannot install certain packages
   (`libatlas-base-dev`, `software-properties-common`, deadsnakes PPA) due to its
   containerized build environment. These packages are required by `reproducer.Dockerfile`.

2. **Build time**: Rebuilding conda, Docker daemon, and system packages from scratch
   takes 5-10 minutes per run, even with Modal's layer caching.

3. **Environment drift**: Package versions change over time. A run today may install
   different versions than a run last week, creating non-deterministic environments
   that make score comparisons unreliable.

Pre-building the base image solves all three issues. The image is built once from the
canonical Dockerfiles, pushed to GHCR, and reused across all runs.

## Consistency with upstream design

This architecture does **not** introduce new abstractions or change the experiment's
semantics. The separation between "shared base" and "task-specific resources" already
exists in the upstream Dockerfiles:

```dockerfile
# Upstream pattern (Alcatraz):
FROM pb-env:latest          # Layer 1: shared base (pre-built locally)
COPY paper.pdf /home/paper/ # Layer 2: task resources (per-task)
```

The Modal runtime makes this same separation explicit using `from_registry()` +
`add_local_file()` instead of `FROM` + `COPY`. The contents are identical.

## Building and pushing images

```bash
cd project/paperbench

# Build pb-env from Dockerfile.base
docker build -f paperbench/Dockerfile.base -t paperbench-env:latest .
docker tag paperbench-env:latest ghcr.io/jzheng17/paperbench-env:latest
docker push ghcr.io/jzheng17/paperbench-env:latest

# Build pb-reproducer from reproducer.Dockerfile
docker build -f paperbench/reproducer.Dockerfile -t paperbench-reproducer:latest .
docker tag paperbench-reproducer:latest ghcr.io/jzheng17/paperbench-reproducer:latest
docker push ghcr.io/jzheng17/paperbench-reproducer:latest
```

## Using the Modal runtime

```bash
# Run BasicAgent on SSC (pilot split) with Modal
python tools/pb_run.py --env-config configs/modal_env.yaml -- \
    paperbench.paper_split=pilot \
    paperbench.solver=paperbench.solvers.basicagent.solver:BasicAgentSolver \
    paperbench.solver.time_limit=3600

# Run OpenHands on SSC with Modal
python tools/pb_run.py --env-config configs/modal_env.yaml -- \
    paperbench.paper_split=pilot \
    paperbench.solver=paperbench.solvers.openhands.solver:OpenHandsSolver \
    paperbench.solver.time_limit=3600
```

## Why personal GHCR instead of upstream's?

The original PaperBench repo does not publish pre-built images to any container
registry. The standard workflow is `docker build` locally, which works for
Alcatraz but not for cloud runtimes (Modal, GKE) that pull from a registry.

The images at `ghcr.io/jzheng17/` are built from the **unmodified upstream
Dockerfiles** (`Dockerfile.base` and `reproducer.Dockerfile`) — they are
byte-identical to local builds. They exist only to enable registry-based
workflows until an official registry is established.

## Image update policy

The pre-built images should be rebuilt and pushed when:
- `Dockerfile.base` or `reproducer.Dockerfile` changes in the upstream repo
- A new parity experiment requires a clean baseline

Images are tagged with `latest` for convenience. For reproducibility, pin to a
specific digest: `ghcr.io/jzheng17/paperbench-env@sha256:...`
