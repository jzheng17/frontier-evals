# Use pb-env by default; override to rebuild from Ubuntu if needed.
ARG BASE_IMAGE=pb-env:latest
FROM ${BASE_IMAGE}

# Set non-interactive mode for apt-get to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

# Update package lists and install common build and ML dependency packages
RUN apt-get update && \
    apt-get install -y \
        software-properties-common \
        wget curl unzip sudo \
        build-essential git cmake \
        libatlas-base-dev libblas-dev liblapack-dev libopenblas-dev \
        gfortran libsm6 libxext6 libxrender-dev && \
    rm -rf /var/lib/apt/lists/*

# Deterministically install both 3.11 and 3.12, users can choose between them
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
        python3.11 python3.11-venv python3.11-dev \
        python3.12 python3.12-venv python3.12-dev \
        python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Set default python version to 3.12
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 2
# users can switch to 3.11 by running `update-alternatives --set python3 /usr/bin/python3.11`

# Ensure python/pip are available for Alcatraz checks.
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# you would then
# 1. make a /submission dir 
# 2. copy the submission there
# 3. run bash /submission/reproduce.sh

# --- hard guarantee: python + pip CLIs exist ---
    RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3-pip python-is-python3 && \
    rm -rf /var/lib/apt/lists/*

# If pip is still missing for any reason, bootstrap it via ensurepip/get-pip.
RUN python3 -m pip --version || ( \
      python3 -m ensurepip --upgrade || true; \
      python3 -m pip --version || ( \
        curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && \
        python3 /tmp/get-pip.py \
      ) \
    )

# Convenience: ensure `pip` exists even if only `pip3` is present
RUN command -v pip || ln -sf "$(command -v pip3)" /usr/local/bin/pip
