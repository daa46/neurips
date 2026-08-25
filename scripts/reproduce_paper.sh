#!/usr/bin/env bash
set -euo pipefail

repository_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repository_root"

python_bin=${PYTHON_BIN:-python3}
output_dir=${1:-results/reproduction}
bootstrap_replicates=${HRMS_BOOTSTRAP:-59}
simulation_repetitions=${HRMS_REPETITIONS:-2}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/hrms-matplotlib-cache}
mkdir -p "$MPLCONFIGDIR" "$output_dir/inputs"

"$python_bin" -m pytest -q
"$python_bin" -m ruff check applications hrms simulations scripts tests

"$python_bin" scripts/generate_synthetic_example.py \
  --output "$output_dir/inputs/synthetic_losses.csv" \
  --registry "$output_dir/inputs/synthetic_model_registry.csv"
cmp "$output_dir/inputs/synthetic_losses.csv" examples/synthetic_losses.csv
cmp "$output_dir/inputs/synthetic_model_registry.csv" examples/synthetic_model_registry.csv

"$python_bin" scripts/run_hrms.py \
  "$output_dir/inputs/synthetic_losses.csv" \
  --model-registry "$output_dir/inputs/synthetic_model_registry.csv" \
  --output "$output_dir/example_hrms" \
  --bootstrap "$bootstrap_replicates" \
  --seed 42

"$python_bin" simulations/run_grid.py \
  --preset smoke \
  --output-dir "$output_dir/simulations" \
  --n-bootstrap "$bootstrap_replicates" \
  --repetitions "$simulation_repetitions" \
  --seed 42

"$python_bin" scripts/make_figures.py \
  --simulations-dir "$output_dir/simulations" \
  --output-dir "$output_dir/figures"

"$python_bin" scripts/write_manifest.py \
  --root "$output_dir" \
  --output "$output_dir/manifest.json" \
  --input canonical_losses=examples/synthetic_losses.csv \
  --input canonical_registry=examples/synthetic_model_registry.csv

echo "Reproduction bundle: $output_dir"
