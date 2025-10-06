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

in_tree_pkg_dir="${build_dir}/tools/torch-mlir/python_packages"
out_of_tree_pkg_dir="${build_dir}/python_packages"

if [[ ! -d "${in_tree_pkg_dir}" && ! -d "${out_of_tree_pkg_dir}" ]]; then
  echo "Couldn't find in-tree or out-of-tree build, exiting."
  exit 1
fi

# The `-nt` check works even if one of the two directories is missing.
if [[ "${in_tree_pkg_dir}" -nt "${out_of_tree_pkg_dir}" ]]; then
  python_packages_dir="${in_tree_pkg_dir}"
else
  python_packages_dir="${out_of_tree_pkg_dir}"
fi

write_env_file() {
  echo "Updating $build_dir/.env file"
  echo "PYTHONPATH=\"$(portable_realpath "$python_packages_dir/torch_mlir")\"" > "$build_dir/.env"
  if ! cp "$build_dir/.env" "$td/.env"; then
    echo "WARNING: Failed to write $td/.env"
  fi
}

write_env_file
