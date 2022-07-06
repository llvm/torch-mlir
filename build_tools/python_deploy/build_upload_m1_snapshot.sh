#!/bin/bash
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# build_upload_m1_snapshot.sh
# This is a wrapper to build_macos_packages.sh to be run on Apple M1 systems
# since GH Actions don't support M1 runners yet and Universal builds
# don't work for Torch-MLIR since we don't have universal PyTorch binaries
# This presumes that dependencies are installed from install_macos_deps.sh and
# you have the gh credentials to upload to the release

set -eu -o errtrace

this_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "$this_dir"/../../ && pwd)"
output_dir="${output_dir:-${this_dir}/wheelhouse}"
rm -rf "${output_dir}"

git fetch --all
latest_snapshot=$(git for-each-ref --sort=creatordate --format '%(refname:short)' refs/tags | tail -n 1)
git checkout "${latest_snapshot}"
git submodule update --init
package_version=${latest_snapshot#"snapshot-"}
echo "Latest snapshot tag is: ${latest_snapshot}"
echo "Latest version is: ${package_version}"

export TORCH_MLIR_PYTHON_VERSIONS="3.9 3.10"
echo "Using Python Versions: ${TORCH_MLIR_PYTHON_VERSIONS}"
export TORCH_MLIR_PYTHON_PACKAGE_VERSION="${package_version}"
echo "Setting torch-mlir Python Package version to: ${TORCH_MLIR_PYTHON_PACKAGE_VERSION}"

TORCH_MLIR_OSX_ARCH=arm64 \
TORCH_MLIR_OSX_TARGET=11.0 \
TORCH_MLIR_PYTHON_PACKAGE_VERSION="${package_version}" \
TORCH_MLIR_PYTHON_VERSIONS="${TORCH_MLIR_PYTHON_VERSIONS}" \
"${repo_root}"/build_tools/python_deploy/build_macos_packages.sh

gh release upload "${latest_snapshot}" "${repo_root}"/build_tools/python_deploy/wheelhouse/torch*.whl
