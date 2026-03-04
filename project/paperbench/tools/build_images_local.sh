#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PLATFORM="${PLATFORM:-linux/arm64}"

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker is not available on PATH." >&2
  exit 1
fi

echo "== Docker environment =="
docker version
docker context show

echo "== Build configuration =="
echo "ROOT_DIR=${ROOT_DIR}"
echo "PLATFORM=${PLATFORM}"

echo "== Checking Dockerfile.base for Miniconda arch =="
if rg -n "Miniconda.*Linux-x86_64" -S "${ROOT_DIR}/paperbench/Dockerfile.base" >/dev/null 2>&1; then
  echo "WARNING: Dockerfile.base still references Linux-x86_64 Miniconda."
  echo "         Apply the arm64 patch before building for Apple Silicon." >&2
fi

echo "== Building pb-env:latest =="
docker build \
  --platform "${PLATFORM}" \
  -t pb-env:latest \
  -f "${ROOT_DIR}/paperbench/Dockerfile.base" \
  "${ROOT_DIR}"

echo "== Building pb-reproducer:latest =="
docker build \
  --platform "${PLATFORM}" \
  -t pb-reproducer:latest \
  -f "${ROOT_DIR}/paperbench/reproducer.Dockerfile" \
  "${ROOT_DIR}"

echo "== Image inspection =="
for image in pb-env:latest pb-reproducer:latest; do
  echo "-- ${image}"
  docker image inspect "${image}" --format 'Architecture={{.Architecture}} OS={{.Os}}'
done
