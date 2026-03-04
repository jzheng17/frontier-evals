#!/usr/bin/env python3
"""Run PaperBench nano entrypoint with injected LocalConfig runtime env."""

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


LOCALCONFIG_TYPE = "alcatraz.clusters.local:LocalConfig"
RUNTIME_TYPE = "nanoeval_alcatraz.alcatraz_computer_interface:AlcatrazComputerRuntime"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wrapper to run paperbench.nano.entrypoint with injected LocalConfig env. "
            "Extra key=value args are forwarded to the nano entrypoint."
        )
    )
    parser.add_argument(
        "--env-config",
        required=True,
        type=Path,
        help="Path to local env YAML (e.g. configs/local_env.yaml)",
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
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def build_env_overrides(component: str, merged: dict[str, Any], overrides: list[str]) -> list[str]:
    injected: list[str] = []
    runtime_prefix = f"paperbench.{component}.computer_runtime"
    env_prefix = f"{runtime_prefix}.env"

    # Runtime override must be an exact key to avoid env.* subkeys blocking runtime injection.
    if not has_exact_key(overrides, runtime_prefix):
        injected.append(f"{runtime_prefix}={RUNTIME_TYPE}")
    else:
        print(f"Skipping {component} runtime injection due to CLI override.")

    if not has_exact_key(overrides, env_prefix):
        injected.append(f"{env_prefix}={LOCALCONFIG_TYPE}")
    else:
        print(f"Skipping {component} env injection due to CLI override.")

    fields = [
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
    for field in fields:
        if field in merged and not has_key_prefix(overrides, f"{env_prefix}.{field}"):
            injected.append(f"{env_prefix}.{field}={stringify_value(merged[field])}")
    return injected


def print_env_summary(component: str, merged: dict[str, Any]) -> None:
    image = merged.get("image")
    pull_from_registry = merged.get("pull_from_registry")
    no_network = merged.get("no_network")
    print(
        f"{component}: image={image} pull_from_registry={pull_from_registry} no_network={no_network}"
    )


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
        merged = merge_component_config(defaults, local_overrides, components[component])
        merged_configs[component] = merged
        injected_overrides.extend(build_env_overrides(component, merged, overrides))

    print("Injected LocalConfig env summary:")
    for component in ("solver", "judge", "reproduction"):
        print_env_summary(component, merged_configs[component])

    argv = [sys.executable, "-m", "paperbench.nano.entrypoint"] + injected_overrides + overrides

    try:
        os.execvpe(argv[0], argv, os.environ)
    except OSError:
        result = subprocess.run(argv, check=False)
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
