#!/usr/bin/env python3
"""Validate experiments/splits/*.txt formatting."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    splits_root = root / "experiments" / "splits"
    split_files = sorted(splits_root.glob("*.txt"))
    if not split_files:
        print("ERROR: No split files found.")
        return 1

    has_error = False
    for split_file in split_files:
        data = split_file.read_bytes()
        if data and not data.endswith(b"\n"):
            print(f"ERROR: Missing trailing newline: {split_file}")
            has_error = True

        lines = split_file.read_text().splitlines()
        for idx, line in enumerate(lines, start=1):
            if not line.strip():
                print(f"ERROR: Empty/whitespace-only line in {split_file}:{idx}")
                has_error = True
            if line != line.strip():
                print(f"ERROR: Leading/trailing whitespace in {split_file}:{idx}")
                has_error = True

    if has_error:
        return 1

    print("OK: split files formatted correctly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
