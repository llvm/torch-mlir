#!/bin/zsh
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Installs Python versions on MacOS necessary to build torch-mlir wheels.
# Downloads official python.org .pkg installers for each requested version.
# This script is intended for CI / release-wheel build environments that use
# the python.org Framework Python. Developers using conda, pyenv, or others
# do NOT need to run this script.
#
# Usage:
#   sudo ./install_macos_deps.sh                                  # installs all supported versions
#   sudo TORCH_MLIR_PYTHON_VERSIONS="3.12" ./install_macos_deps.sh  # single version
#
# The TORCH_MLIR_PYTHON_VERSIONS variable should match what you pass to
# build_macos_packages.sh. When new Python patch releases come out, update
# the PYTHON_URL_MAP below (check https://www.python.org/downloads/).

set -eu -o pipefail

if [[ "$(whoami)" != "root" ]]; then
  echo "ERROR: Must setup deps as root"
  exit 1
fi

# Which Python minor versions to install. Keep in sync with build_macos_packages.sh defaults.
TORCH_MLIR_PYTHON_VERSIONS="${TORCH_MLIR_PYTHON_VERSIONS:-3.10 3.11 3.12}"

# Map of minor version -> latest known patch installer URL.
# Update these URLs when new patch releases are published on python.org.
typeset -A PYTHON_URL_MAP
PYTHON_URL_MAP=(
  [3.10]="https://www.python.org/ftp/python/3.10.11/python-3.10.11-macos11.pkg"
  [3.11]="https://www.python.org/ftp/python/3.11.9/python-3.11.9-macos11.pkg"
  [3.12]="https://www.python.org/ftp/python/3.12.10/python-3.12.10-macos11.pkg"
)

# ${=...} forces word splitting in zsh (not done by default unlike bash).
for python_version in ${=TORCH_MLIR_PYTHON_VERSIONS}; do
  url="${PYTHON_URL_MAP[$python_version]}"
  if [[ -z "$url" ]]; then
    echo "ERROR: No installer URL configured for Python $python_version"
    echo "       Add an entry to PYTHON_URL_MAP in this script."
    exit 1
  fi

  echo "-- Installing Python $python_version from $url"
  python_path="/Library/Frameworks/Python.framework/Versions/$python_version"
  python_exe="$python_path/bin/python3"

  # Install Python.
  if ! [ -x "$python_exe" ]; then
    package_basename="$(basename "$url")"
    download_path="/tmp/torch_mlir_python_install/$package_basename"
    mkdir -p "$(dirname "$download_path")"
    echo "Downloading $url -> $download_path"
    curl -L "$url" -o "$download_path"

    echo "Installing $download_path"
    installer -pkg "$download_path" -target /
    rm -f "$download_path"
  else
    echo ":: Python $python_version already installed at $python_exe. Not reinstalling."
  fi

  echo ":: Python version $python_version installed:"
  $python_exe --version
  $python_exe -m pip --version

  echo ":: Installing pip packages"
  $python_exe -m pip install --upgrade pip
  $python_exe -m pip install --upgrade delocate
done

echo "*** All done ***"
