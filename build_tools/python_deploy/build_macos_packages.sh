#!/bin/bash
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# build_macos_packages.sh
# One stop build of torch-mlir Python packages for MacOS. This presumes that
# dependencies are installed from install_macos_deps.sh. This will build
# for a list of Python versions synchronized with that script and corresponding
# with directory names under:
#   /Library/Frameworks/Python.framework/Versions
#
# MacOS convention is to refer to this as major.minor (i.e. "3.9", "3.10").
# Valid packages:
#   torch-mlir

set -eu -o errtrace

this_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "$this_dir"/../../ && pwd)"
python_versions="${TORCH_MLIR_PYTHON_VERSIONS:-3.9 3.10 3.11}"
output_dir="${output_dir:-${this_dir}/wheelhouse}"
packages="${packages:-torch-mlir}"

PKG_VER_FILE="${repo_root}"/torch_mlir_package_version ; [ -f "$PKG_VER_FILE" ] && . "$PKG_VER_FILE"
export TORCH_MLIR_PYTHON_PACKAGE_VERSION="${TORCH_MLIR_PYTHON_PACKAGE_VERSION:-0.0.1}"
echo "Setting torch-mlir Python Package version to: ${TORCH_MLIR_PYTHON_PACKAGE_VERSION}"

# Note that this typically is selected to match the version that the official
# Python distributed is built at.
export MACOSX_DEPLOYMENT_TARGET="${TORCH_MLIR_OSX_TARGET:-11.1}"
export CMAKE_OSX_ARCHITECTURES="${TORCH_MLIR_OSX_ARCH:-arm64;x86_64}"
echo "CMAKE_OSX_ARCHITECTURES: $CMAKE_OSX_ARCHITECTURES"
echo "MACOSX_DEPLOYMENT_TARGET $MACOSX_DEPLOYMENT_TARGET"

# Disable LTC build on MacOS to avoid linkage issues
# https://github.com/llvm/torch-mlir/issues/1253
export TORCH_MLIR_ENABLE_LTC=0

function run() {
  echo "Using python versions: ${python_versions}"

  local orig_path="$PATH"

  # Build phase.
  for package in $packages; do
    echo "******************** BUILDING PACKAGE ${package} ********************"
    for python_version in $python_versions; do
      python_dir="/Library/Frameworks/Python.framework/Versions/$python_version"
      if ! [ -x "$python_dir/bin/python3" ]; then
        echo "ERROR: Could not find python3: $python_dir (skipping)"
        continue
      fi
      export PATH=$python_dir/bin:$orig_path
      echo ":::: Python version $(python3 --version)"
      case "$package" in
        torch-mlir-ext)
          clean_wheels torch_mlir_ext "$python_version"
          build_torch_mlir_ext torch_mlir_ext "$python_version"
          run_audit_wheel torch_mlir_ext "$python_version"
          ;;
        torch-mlir)
          clean_wheels torch_mlir "$python_version"
          build_torch_mlir torch_mlir "$python_version"
          run_audit_wheel torch_mlir "$python_version"
          ;;
        *)
          echo "Unrecognized package '$package'"
          exit 1
          ;;
      esac
    done
  done
}

function build_torch_mlir_ext() {
  local wheel_basename="$1"
  local python_version="$2"
  rm -rf "$output_dir"/build_venv
  python"${python_version}" -m venv "$output_dir"/build_venv
  source "$output_dir"/build_venv/bin/activate
  python"${python_version}" -m pip install -U pip
  python"${python_version}" -m pip install -r "$repo_root"/pytorch-requirements.txt --extra-index-url https://download.pytorch.org/whl/nightly/cpu
  python"${python_version}" -m pip install -r "$repo_root"/build-requirements.txt
  CMAKE_GENERATOR=Ninja \
  TORCH_MLIR_PYTHON_PACKAGE_VERSION=${TORCH_MLIR_PYTHON_PACKAGE_VERSION} \
  MACOSX_DEPLOYMENT_TARGET=$MACOSX_DEPLOYMENT_TARGET \
  CMAKE_OSX_ARCHITECTURES=$CMAKE_OSX_ARCHITECTURES \
  python"${python_version}" -m pip wheel -v --no-build-isolation -w "$output_dir" "$repo_root" --extra-index-url https://download.pytorch.org/whl/nightly/cpu
  deactivate
  rm -rf "$output_dir"/build_venv
}

function build_torch_mlir() {
  local wheel_basename="$1"
  local python_version="$2"
  rm -rf "$output_dir"/build_venv
  python"${python_version}" -m venv "$output_dir"/build_venv
  source "$output_dir"/build_venv/bin/activate
  python"${python_version}" -m pip install -U pip delocate
  python"${python_version}" -m pip install -r "$repo_root"/build-requirements.txt
  CMAKE_GENERATOR=Ninja \
  TORCH_MLIR_PYTHON_PACKAGE_VERSION=${TORCH_MLIR_PYTHON_PACKAGE_VERSION} \
  MACOSX_DEPLOYMENT_TARGET=$MACOSX_DEPLOYMENT_TARGET \
  CMAKE_OSX_ARCHITECTURES=$CMAKE_OSX_ARCHITECTURES \
  TORCH_MLIR_ENABLE_JIT_IR_IMPORTER=0 \
  TORCH_MLIR_ENABLE_ONLY_MLIR_PYTHON_BINDINGS=1 \
  python"${python_version}" -m pip wheel -v --no-build-isolation -w "$output_dir" "$repo_root"
  deactivate
  rm -rf "$output_dir"/build_venv
}

function clean_wheels() {
  local wheel_basename="$1"
  local python_version="$2"
  echo ":::: Clean wheels $wheel_basename $python_version"
  rm -rf "$repo_root"/build/
  rm -f "$output_dir"/"${wheel_basename}"-*-"${python_version//./}"-*.whl
}

function run_audit_wheel() {
  set +x
  local wheel_basename="$1"
  local python_version="$2"
  generic_wheel=$(ls "$output_dir"/"${wheel_basename}"-* | grep "${python_version//./}")
  echo "Looking for $generic_wheel"
  if [ -f "$generic_wheel" ]; then
    echo "$generic_wheel found. Delocating it.."
    rm -rf "$output_dir"/test_venv
    python"${python_version}" -m venv "$output_dir"/test_venv
    source "$output_dir"/test_venv/bin/activate
    python"${python_version}" -m pip install -U pip
    python"${python_version}" -m pip install -r "$repo_root"/pytorch-requirements.txt --extra-index-url https://download.pytorch.org/whl/nightly/cpu
    python"${python_version}" -m pip install -r "$repo_root"/build-requirements.txt
    python"${python_version}" -m pip install "$generic_wheel" --extra-index-url https://download.pytorch.org/whl/nightly/cpu
    DYLD_LIBRARY_PATH="$output_dir"/test_venv/lib/python"${python_version}"/site-packages/torch/lib delocate-wheel -v "$generic_wheel"
    deactivate
    rm -rf "$output_dir"/test_venv
  fi
}

run
