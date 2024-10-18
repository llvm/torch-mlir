#!/bin/bash
set -euo pipefail

src_dir="$(realpath "$(dirname "$0")"/..)"
project_dir="$src_dir/../.."

cd "$src_dir"

# Ensure PYTHONPATH is set for export to child processes, even if empty.
export PYTHONPATH=${PYTHONPATH-}
python -m e2e_testing.main "$@"
