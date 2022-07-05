#!/bin/bash
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# build_linux_packages.sh
# One stop build of IREE Python packages for Linux. The Linux build is
# complicated because it has to be done via a docker container that has
# an LTS glibc version, all Python packages and other deps.
# This script handles all of those details.
#
# Usage:
# Build everything (all packages, all python versions):
#   ./build_tools/python_deploy/build_linux_packages.sh
#
# Build specific Python versions and packages to custom directory:
#   python_versions="cp38-cp38 cp39-cp39" \
#   packages="torch-mlir" \
#   output_dir="/tmp/wheelhouse" \
#   ./build_tools/python_deploy/build_linux_packages.sh
#
# Valid Python versions match a subdirectory under /opt/python in the docker
# image. Typically:
#   cp37-cp37m cp38-cp38 cp39-cp39 cp310-cp310
#
# Valid packages:
#   torch-mlir
#
# Note that this script is meant to be run on CI and it will pollute both the
# output directory and in-tree build/ directories (under runtime/ and
# iree/compiler/) with docker created, root owned builds. Sorry - there is
# no good way around it.
#
# It can be run on a workstation but recommend using a git worktree dedicated
# to packaging to avoid stomping on development artifacts.
set -eu -o errtrace

this_dir="$(cd $(dirname $0) && pwd)"
script_name="$(basename $0)"
repo_root="$(cd $this_dir/../../ && pwd)"
script_name="$(basename $0)"
#manylinux_docker_image="${manylinux_docker_image:-stellaraccident/manylinux2014_x86_64-bazel-5.1.0:latest}"
manylinux_docker_image="${manylinux_docker_image:-quay.io/pypa/manylinux_2_28_x86_64:latest}"
python_versions="${TM_PYTHON_VERSIONS:-cp39-cp39 cp310-cp310 cp311-cp311}"
output_dir="${output_dir:-${this_dir}/wheelhouse}"
packages="${packages:-torch-mlir}"

PKG_VER_FILE=${repo_root}/torch_mlir_package_version ; [ -f $PKG_VER_FILE ] && . $PKG_VER_FILE
export TORCH_MLIR_PYTHON_PACKAGE_VERSION="${TORCH_MLIR_PYTHON_PACKAGE_VERSION:-0.0.1}"
echo "Setting torch-mlir Python Package version to: ${TORCH_MLIR_PYTHON_PACKAGE_VERSION}"

function run_on_host() {
  echo "Running on host"
  echo "Launching docker image ${manylinux_docker_image}"
  echo "Outputting to ${output_dir}"
  rm -rf "${output_dir}"
  mkdir -p "${output_dir}"
  docker run --rm \
    -v "${repo_root}:/main_checkout/torch-mlir" \
    -v "${output_dir}:/wheelhouse" \
    -e __MANYLINUX_BUILD_WHEELS_IN_DOCKER=1 \
    -e "TORCH_MLIR_PYTHON_PACKAGE_VERSION=${TORCH_MLIR_PYTHON_PACKAGE_VERSION}" \
    -e "TM_PYTHON_VERSIONS=${python_versions}" \
    -e "packages=${packages}" \
    ${manylinux_docker_image} \
    -- bash /main_checkout/torch-mlir/build_tools/python_deploy/build_linux_packages.sh
}

function run_in_docker() {
  echo "Running in docker"
  echo "Using python versions: ${python_versions}"

  local orig_path="$PATH"

  # Build phase.
  for package in $packages; do
    echo "******************** BUILDING PACKAGE ${package} ********************"
    for python_version in $python_versions; do
      python_dir="/opt/python/$python_version"
      if ! [ -x "$python_dir/bin/python" ]; then
        echo "ERROR: Could not find python: $python_dir (skipping)"
        continue
      fi
      export PATH=$python_dir/bin:$orig_path
      echo ":::: Python version $(python --version)"
      case "$package" in
        torch-mlir)
          install_clang
          clean_wheels torch_mlir $python_version
          build_torch_mlir $python_version
          #run_audit_wheel torch_mlir $python_version
          ;;
        *)
          echo "Unrecognized package '$package'"
          exit 1
          ;;
      esac
    done
  done
}

function install_clang() {
  yum install -y llvm-toolset
  clang --version
}

function build_torch_mlir() {
  local python_version="$1"
  git config --global --add safe.directory /main_checkout/torch-mlir/externals/pytorch
  /opt/python/${python_version}/bin/pip install -r /main_checkout/torch-mlir/requirements.txt
  CMAKE_GENERATOR=Ninja \
  TORCH_MLIR_PYTHON_PACKAGE_VERSION=${TORCH_MLIR_PYTHON_PACKAGE_VERSION} \
  TORCH_MLIR_PYTHON_VERSION=/opt/python/${python_version}/bin/python \
  TORCH_MLIR_PIP_VERSION=/opt/python/${python_version}/bin/pip \
  CC=`which clang` CXX=`which clang++` \
  python -m pip wheel -v -w /wheelhouse /main_checkout/torch-mlir/
}

function run_audit_wheel() {
  local wheel_basename="$1"
  local python_version="$2"
  generic_wheel="/wheelhouse/${wheel_basename}-*-${python_version}-linux_x86_64.whl"
  echo ":::: Auditwheel $generic_wheel"
  auditwheel repair -w /wheelhouse $generic_wheel
  rm $generic_wheel
}

function clean_wheels() {
  local wheel_basename="$1"
  local python_version="$2"
  echo ":::: Clean wheels $wheel_basename $python_version"
  rm -f /wheelhouse/${wheel_basename}-*-${python_version}-*.whl
}

# Trampoline to the docker container if running on the host.
if [ -z "${__MANYLINUX_BUILD_WHEELS_IN_DOCKER-}" ]; then
  run_on_host "$@"
else
  run_in_docker "$@"
fi
