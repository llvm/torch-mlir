#!/bin/bash
# generate the .env file with default options.
#
# For arbitrary build/install directories, set the env variables:
# - TORCH_MLIR_BUILD_DIR

set -eu -o pipefail

portable_realpath() {
  # Create the directory if needed so that the `cd` doesn't fail.
  mkdir -p "$1" && cd "$1" && pwd
}

td="$(portable_realpath "$(dirname "$0")"/..)"
build_dir="$(portable_realpath "${TORCH_MLIR_BUILD_DIR:-$td/build}")"
python_packages_dir="$build_dir/python_packages"

write_env_file() {
  echo "Updating $build_dir/.env file"
  echo "PYTHONPATH=\"$(portable_realpath "$python_packages_dir/torch_mlir")\"" > "$build_dir/.env"
  if ! cp "$build_dir/.env" "$td/.env"; then
    echo "WARNING: Failed to write $td/.env"
  fi
}

write_env_file
