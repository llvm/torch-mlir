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
# MacOS convention is to refer to this as major.minor (i.e. "3.10", "3.11", "3.12").
# Supports both x86_64 and arm64 (Apple Silicon) natively — the build
# auto-detects the host architecture via CMAKE_OSX_ARCHITECTURES.
#
# Valid packages:
#   torch-mlir

set -eu -o errtrace

this_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "$this_dir"/../../ && pwd)"
# Officially supported Python minor versions for macOS wheel builds.
SUPPORTED_PYTHON_VERSIONS="3.10 3.11 3.12"

# If TORCH_MLIR_PYTHON_VERSIONS is not set, auto-detect from the currently
# active python3 (i.e. whatever conda/pyenv/brew has made active on PATH).
# For multi-version release builds, set the variable explicitly, e.g.:
#   TORCH_MLIR_PYTHON_VERSIONS="3.10 3.11 3.12" ./build_macos_packages.sh
if [[ -n "${TORCH_MLIR_PYTHON_VERSIONS:-}" ]]; then
  python_versions="${TORCH_MLIR_PYTHON_VERSIONS}"
else
  python_versions=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null) || {
    echo "ERROR: TORCH_MLIR_PYTHON_VERSIONS not set and python not found on PATH."
    exit 1
  }
  if [[ " ${SUPPORTED_PYTHON_VERSIONS} " != *" ${python_versions} "* ]]; then
    echo "WARNING: Auto-detected Python ${python_versions} is not in the supported set (${SUPPORTED_PYTHON_VERSIONS})."
    echo "         Set TORCH_MLIR_PYTHON_VERSIONS explicitly to suppress this warning."
  fi
  echo "Auto-detected Python version: ${python_versions}"
fi
output_dir="${output_dir:-${this_dir}/wheelhouse}"
packages="${packages:-torch-mlir}"

PKG_VER_FILE="${repo_root}"/torch_mlir_package_version ; [ -f "$PKG_VER_FILE" ] && . "$PKG_VER_FILE"
export TORCH_MLIR_PYTHON_PACKAGE_VERSION="${TORCH_MLIR_PYTHON_PACKAGE_VERSION:-0.0.1}"
echo "Setting torch-mlir Python Package version to: ${TORCH_MLIR_PYTHON_PACKAGE_VERSION}"

# Note that this typically is selected to match the version that the official
# Python distributed is built at.
export MACOSX_DEPLOYMENT_TARGET="${TORCH_MLIR_OSX_TARGET:-11.1}"
export CMAKE_OSX_ARCHITECTURES="${TORCH_MLIR_OSX_ARCH:-$(uname -m)}"
echo "CMAKE_OSX_ARCHITECTURES: $CMAKE_OSX_ARCHITECTURES"
echo "MACOSX_DEPLOYMENT_TARGET $MACOSX_DEPLOYMENT_TARGET"

# LTC is disabled for release wheel builds to align with the Linux CI release
# default. The earlier macOS linking issue (#1253) has been fixed, but LTC
# remains off for releases consistent with build_linux_packages.sh.

function run() {
  echo "Using python versions: ${python_versions}"

  local orig_path="$PATH"

  # Build phase.
  for package in $packages; do
    echo "******************** BUILDING PACKAGE ${package} ********************"
    for python_version in $python_versions; do
      # Prefer python{X.Y} already on PATH (conda, pyenv, brew, etc.).
      # The user is responsible for activating the right environment beforehand.
      # Falls back to the official python.org framework installer path.
      local python_dir
      if python_exe=$(command -v "python${python_version}" 2>/dev/null); then
        python_dir="$(dirname "$python_exe")"
      elif [ -x "/Library/Frameworks/Python.framework/Versions/${python_version}/bin/python3" ]; then
        python_dir="/Library/Frameworks/Python.framework/Versions/${python_version}/bin"
      else
        echo "ERROR: python${python_version} not found on PATH and not in Python.framework (skipping)"
        echo "       Activate the right environment first (e.g. conda activate, pyenv local) or run install_macos_deps.sh"
        continue
      fi
      export PATH=$python_dir:$orig_path
      echo ":::: Python version $(python${python_version} --version)"
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
  # Disable LTC build for releases (aligns with Linux CI release default).
  export TORCH_MLIR_ENABLE_LTC=0
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
  # Disable LTC build for releases (aligns with Linux CI release default).
  export TORCH_MLIR_ENABLE_LTC=0
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
    python"${python_version}" -m pip install -U pip delocate
    python"${python_version}" -m pip install -r "$repo_root"/pytorch-requirements.txt --extra-index-url https://download.pytorch.org/whl/nightly/cpu
    python"${python_version}" -m pip install -r "$repo_root"/build-requirements.txt
    python"${python_version}" -m pip install "$generic_wheel" --extra-index-url https://download.pytorch.org/whl/nightly/cpu
    DYLD_LIBRARY_PATH="$output_dir"/test_venv/lib/python"${python_version}"/site-packages/torch/lib delocate-wheel -v "$generic_wheel"
    deactivate
    rm -rf "$output_dir"/test_venv
  fi
}

run
