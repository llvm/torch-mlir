#!/bin/bash
set -euo pipefail

src_dir="$(realpath "$(dirname "$0")"/..)"
build_dir="${src_dir}/build"

export INCLUDE_CUSTOM_OP="${INCLUDE_CUSTOM_OP:-OFF}"

cd "$src_dir"

# Ensure PYTHONPATH is set for export to child processes, even if empty.
export PYTHONPATH=${PYTHONPATH-}
source .env

python -m e2e_testing.main "$@"
