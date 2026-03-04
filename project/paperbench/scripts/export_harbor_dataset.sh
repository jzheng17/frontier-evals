#!/usr/bin/env bash
set -euo pipefail

# Export dev split into the default output directory.
python scripts/export_harbor_dataset.py --split dev --clean

# Export a single paper.
python scripts/export_harbor_dataset.py --papers semantic-self-consistency --clean

# Materialize the pilot reference submission.
python scripts/materialize_pilot_reference.py --out ./_harbor_export
