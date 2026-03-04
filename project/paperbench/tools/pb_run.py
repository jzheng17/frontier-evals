#!/usr/bin/env python3
"""Run PaperBench nano entrypoint with injected runtime env.

Supports two runtime backends:
  - Alcatraz/LocalConfig (local Docker via docker.sock)
  - Modal (Modal Sandboxes with GPU support)

The backend is determined by the `type` field in each component's config
within the env YAML file.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


LOCALCONFIG_TYPE = "alcatraz.clusters.local:LocalConfig"
ALCATRAZ_RUNTIME_TYPE = (
    "nanoeval_alcatraz.alcatraz_computer_interface:AlcatrazComputerRuntime"
)
MODAL_RUNTIME_TYPE = "paperbench.runtime.modal:ModalComputerRuntime"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wrapper to run paperbench.nano.entrypoint with injected runtime env. "
            "Extra key=value args are forwarded to the nano entrypoint."
        )
    )
    parser.add_argument(
        "--env-config",
        required=True,
        type=Path,
        help="Path to env YAML (e.g. configs/local_env.yaml or configs/modal_env.yaml)",
    )
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Forwarded key=value overrides for paperbench.nano.entrypoint",
    )
    return parser.parse_args()


def load_env_yaml(path: Path) -> dict[str, Any]:
    from paperbench.utils import load_yaml_dict

    return load_yaml_dict(path)


def merge_component_config(
    defaults: dict[str, Any],
    local_overrides: dict[str, Any],
    component: dict[str, Any],
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for source in (defaults, local_overrides, component):
        for key, value in source.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value
    return merged


def has_exact_key(overrides: list[str], key: str) -> bool:
    for arg in overrides:
        if arg.startswith(key + "="):
            return True
    return False


def has_key_prefix(overrides: list[str], prefix: str) -> bool:
    for arg in overrides:
        if arg.startswith(prefix + "=") or arg.startswith(prefix + "."):
            return True
    return False


def stringify_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def _is_modal_runtime(runtime_type: str) -> bool:
    return "modal" in runtime_type.lower() or "Modal" in runtime_type


# -- Alcatraz-specific injection --

ALCATRAZ_FIELDS = [
    "image",
    "pull_from_registry",
    "no_network",
    "is_nvidia_gpu_env",
    "docker_host",
    "local_network",
    "volumes_config",
    "environment",
    "side_images",
    "jupyter_setup",
    "wait_for_health",
]


def build_alcatraz_overrides(
    component: str, merged: dict[str, Any], overrides: list[str]
) -> list[str]:
    injected: list[str] = []
    runtime_prefix = f"paperbench.{component}.computer_runtime"
    env_prefix = f"{runtime_prefix}.env"

    if not has_exact_key(overrides, runtime_prefix):
        injected.append(f"{runtime_prefix}={ALCATRAZ_RUNTIME_TYPE}")

    if not has_exact_key(overrides, env_prefix):
        injected.append(f"{env_prefix}={LOCALCONFIG_TYPE}")

    for field in ALCATRAZ_FIELDS:
        if field in merged and not has_key_prefix(overrides, f"{env_prefix}.{field}"):
            injected.append(
                f"{env_prefix}.{field}={stringify_value(merged[field])}"
            )
    return injected


# -- Modal-specific injection --

MODAL_FIELDS = [
    "dockerfile",
    "context_dir",
    "image_tag",
    "gpu_type",
    "gpu_count",
    "sandbox_timeout",
    "app_name",
    "block_network",
    "environment",
    "enable_docker_daemon",
]


def build_modal_overrides(
    component: str, merged: dict[str, Any], overrides: list[str]
) -> list[str]:
    injected: list[str] = []
    runtime_prefix = f"paperbench.{component}.computer_runtime"

    # Inject the runtime class itself.
    if not has_exact_key(overrides, runtime_prefix):
        injected.append(f"{runtime_prefix}={MODAL_RUNTIME_TYPE}")

    # Inject fields directly on the runtime (no .env intermediary).
    for field in MODAL_FIELDS:
        if field in merged and not has_key_prefix(
            overrides, f"{runtime_prefix}.{field}"
        ):
            injected.append(
                f"{runtime_prefix}.{field}={stringify_value(merged[field])}"
            )
    return injected


# -- Unified dispatch --


def build_env_overrides(
    component: str, merged: dict[str, Any], overrides: list[str]
) -> list[str]:
    runtime_type = merged.get("type", ALCATRAZ_RUNTIME_TYPE)
    if _is_modal_runtime(runtime_type):
        return build_modal_overrides(component, merged, overrides)
    else:
        return build_alcatraz_overrides(component, merged, overrides)


def print_env_summary(component: str, merged: dict[str, Any]) -> None:
    runtime_type = merged.get("type", "alcatraz")
    if _is_modal_runtime(runtime_type):
        dockerfile = merged.get("dockerfile", "?")
        gpu = merged.get("gpu_type", "none")
        gpu_count = merged.get("gpu_count", 0)
        print(f"  {component}: modal  dockerfile={dockerfile}  gpu={gpu}:{gpu_count}")
    else:
        image = merged.get("image", "?")
        gpu = merged.get("is_nvidia_gpu_env", False)
        print(f"  {component}: alcatraz  image={image}  gpu={gpu}")


def main() -> None:
    args = parse_args()
    overrides = list(args.overrides)
    if overrides and overrides[0] == "--":
        overrides = overrides[1:]

    env_yaml = load_env_yaml(args.env_config)
    defaults = env_yaml.get("defaults", {}) or {}
    local_overrides = env_yaml.get("local", {}) or {}
    components = env_yaml.get("components", {}) or {}

    injected_overrides: list[str] = []
    merged_configs: dict[str, dict[str, Any]] = {}

    for component in ("solver", "judge", "reproduction"):
        if component not in components:
            raise ValueError(f"Missing components.{component} in env config.")
        merged = merge_component_config(
            defaults, local_overrides, components[component]
        )
        merged_configs[component] = merged
        injected_overrides.extend(build_env_overrides(component, merged, overrides))

    # Detect runtime backend from first component.
    first_type = merged_configs["solver"].get("type", "alcatraz")
    backend = "Modal" if _is_modal_runtime(first_type) else "Alcatraz/LocalConfig"
    print(f"Runtime backend: {backend}")
    for component in ("solver", "judge", "reproduction"):
        print_env_summary(component, merged_configs[component])

    argv = (
        [sys.executable, "-m", "paperbench.nano.entrypoint"]
        + injected_overrides
        + overrides
    )

    try:
        os.execvpe(argv[0], argv, os.environ)
    except OSError:
        result = subprocess.run(argv, check=False)
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
