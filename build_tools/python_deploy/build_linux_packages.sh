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
#   TM_PYTHON_VERSIONS="cp38-cp38 cp39-cp39" \
#   TM_PACKAGES="torch-mlir" \
#   TM_OUTPUT_DIR="/tmp/wheelhouse" \
#   ./build_tools/python_deploy/build_linux_packages.sh
#
# Valid Python versions match a subdirectory under /opt/python in the docker
# image. Typically:
#   cp39-cp39 cp310-cp310
#
# Valid packages:
#   torch-mlir, in-tree, out-of-tree
#
# Note that this script is meant to be run on CI and it will pollute both the
# output directory and in-tree build/ directories with docker created, root owned builds.
# Sorry - there is no good way around it but TODO: move to using user UID/GID.
#
# It can be run on a workstation but recommend using a git worktree dedicated
# to packaging to avoid stomping on development artifacts.
set -eu -o errtrace

this_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "$this_dir"/../../ && pwd)"
arch="$(uname -m)"
echo "Running on Arch: ${arch}"
# This needs to be a manylinux image so we can ship pip packages
TM_RELEASE_DOCKER_IMAGE="${TM_RELEASE_DOCKER_IMAGE:-quay.io/pypa/manylinux2014_${arch}}"
# This assumes an Ubuntu LTS like image. You can build your own with
# ./build_tools/docker/Dockerfile
TM_CI_DOCKER_IMAGE="${TM_CI_DOCKER_IMAGE:-powderluv/torch-mlir-ci:latest}"
# Version of Python to use in Release builds. Ignored in CIs.
TM_PYTHON_VERSIONS="${TM_PYTHON_VERSIONS:-cp38-cp38 cp310-cp310 cp311-cp311}"
# Location to store Release wheels
TM_OUTPUT_DIR="${TM_OUTPUT_DIR:-${this_dir}/wheelhouse}"
# What "packages to build"
TM_PACKAGES="${TM_PACKAGES:-torch-mlir torch-mlir-core}"
# Use pre-built Pytorch
TM_USE_PYTORCH_BINARY="${TM_USE_PYTORCH_BINARY:-ON}"
# Skip running tests if you want quick iteration
TM_SKIP_TESTS="${TM_SKIP_TESTS:-OFF}"
# Update ODS and abstract interpretation library files
TM_UPDATE_ODS_AND_ABSTRACT_INTERP_LIB="${TM_UPDATE_ODS_AND_ABSTRACT_INTERP_LIB:-OFF}"
# Determine if we use a stable or a nightly torch build
TM_TORCH_VERSION="${TM_TORCH_VERSION:-nightly}"

PKG_VER_FILE="${repo_root}"/torch_mlir_package_version ; [ -f "$PKG_VER_FILE" ] && . "$PKG_VER_FILE"
TORCH_MLIR_PYTHON_PACKAGE_VERSION="${TORCH_MLIR_PYTHON_PACKAGE_VERSION:-0.0.1}"
echo "Setting torch-mlir Python Package version to: ${TORCH_MLIR_PYTHON_PACKAGE_VERSION}"

export TORCH_MLIR_SRC_PYTORCH_REPO="${TORCH_MLIR_SRC_PYTORCH_REPO:-pytorch/pytorch}"
echo "Setting torch-mlir PyTorch Repo for source builds to: ${TORCH_MLIR_SRC_PYTORCH_REPO}"
export TORCH_MLIR_SRC_PYTORCH_BRANCH="${TORCH_MLIR_SRC_PYTORCH_BRANCH:-master}"
echo "Setting torch-mlir PyTorch version for source builds to: ${TORCH_MLIR_SRC_PYTORCH_BRANCH}"

# If using PyTorch source, install from the existing build instead of rebuilding
# all of PyTorch.  This option is useful in CI, when it determines that the
# PyTorch version has not changed between consecutive runs.
export TM_PYTORCH_INSTALL_WITHOUT_REBUILD="${TM_PYTORCH_INSTALL_WITHOUT_REBUILD:-false}"

function run_on_host() {
  echo "Running on host for $1:$@"
  echo "Outputting to ${TM_OUTPUT_DIR}"
  if [[ $TM_PYTORCH_INSTALL_WITHOUT_REBUILD != "true" ]]; then
    # We want to use the cached files, so don't remove them.
    rm -rf "${TM_OUTPUT_DIR}"
  fi
  mkdir -p "${TM_OUTPUT_DIR}"
  case "$package" in
    torch-mlir)
      TM_CURRENT_DOCKER_IMAGE=${TM_RELEASE_DOCKER_IMAGE}
      export USERID=0
      export GROUPID=0
      ;;
    torch-mlir-core)
      TM_CURRENT_DOCKER_IMAGE=${TM_RELEASE_DOCKER_IMAGE}
      export USERID=0
      export GROUPID=0
      ;;
    out-of-tree)
      TM_CURRENT_DOCKER_IMAGE=${TM_CI_DOCKER_IMAGE}
      # CI uses only Python3.10
      TM_PYTHON_VERSIONS="cp310-cp310"
      export USERID=$(id -u)
      export GROUPID=$(id -g)
      ;;
    in-tree)
      TM_CURRENT_DOCKER_IMAGE=${TM_CI_DOCKER_IMAGE}
      # CI uses only Python3.10
      TM_PYTHON_VERSIONS="cp310-cp310"
      export USERID=$(id -u)
      export GROUPID=$(id -g)
      ;;
    *)
      echo "Unrecognized package '$package'"
      exit 1
      ;;
  esac
  echo "Launching docker image ${TM_CURRENT_DOCKER_IMAGE} with UID:${USERID} GID:${GROUPID}"
  docker run --rm \
    -v "${repo_root}:/main_checkout/torch-mlir" \
    -v "${TM_OUTPUT_DIR}:/wheelhouse" \
    -v "${HOME}:/home/${USER}" \
    --user ${USERID}:${GROUPID} \
    --workdir="/home/$USER" \
    --volume="/etc/group:/etc/group:ro" \
    --volume="/etc/passwd:/etc/passwd:ro" \
    --volume="/etc/shadow:/etc/shadow:ro" \
    --ipc=host \
    -e __MANYLINUX_BUILD_WHEELS_IN_DOCKER=1 \
    -e "TORCH_MLIR_PYTHON_PACKAGE_VERSION=${TORCH_MLIR_PYTHON_PACKAGE_VERSION}" \
    -e "TM_PYTHON_VERSIONS=${TM_PYTHON_VERSIONS}" \
    -e "TM_PACKAGES=${package}" \
    -e "TM_SKIP_TESTS=${TM_SKIP_TESTS}" \
    -e "TM_UPDATE_ODS_AND_ABSTRACT_INTERP_LIB=${TM_UPDATE_ODS_AND_ABSTRACT_INTERP_LIB}" \
    -e "TM_USE_PYTORCH_BINARY=${TM_USE_PYTORCH_BINARY}" \
    -e "TORCH_MLIR_SRC_PYTORCH_REPO=${TORCH_MLIR_SRC_PYTORCH_REPO}" \
    -e "TORCH_MLIR_SRC_PYTORCH_BRANCH=${TORCH_MLIR_SRC_PYTORCH_BRANCH}" \
    -e "TM_PYTORCH_INSTALL_WITHOUT_REBUILD=${TM_PYTORCH_INSTALL_WITHOUT_REBUILD}" \
    -e "TM_TORCH_VERSION=${TM_TORCH_VERSION}" \
    -e "CCACHE_DIR=/main_checkout/torch-mlir/.ccache" \
    "${TM_CURRENT_DOCKER_IMAGE}" \
    /bin/bash /main_checkout/torch-mlir/build_tools/python_deploy/build_linux_packages.sh
}

function run_in_docker() {
  echo "Running in docker"
  echo "Using python versions: ${TM_PYTHON_VERSIONS}"

  local orig_path="$PATH"

  # Build phase.
  for package in $TM_PACKAGES; do
    echo "******************** BUILDING PACKAGE ${package} (docker) ************"
    for python_version in $TM_PYTHON_VERSIONS; do
      python_dir="/opt/python/$python_version"
      if ! [ -x "$python_dir/bin/python" ]; then
        echo "Could not find python: $python_dir (using system default Python3)"
	      python_dir=`which python3`
        echo "Defaulting to $python_dir (expected for CI builds)"
      fi
      export PATH=$python_dir/bin:$orig_path
      echo ":::: Python version $(python3 --version)"
      case "$package" in
        torch-mlir)
          clean_wheels torch_mlir "$python_version"
          build_torch_mlir "$TM_TORCH_VERSION"

          # Disable audit wheel until we can fix ODR torch issues.  See
          # https://github.com/llvm/torch-mlir/issues/1709
          #
          #run_audit_wheel torch_mlir "$python_version"

          clean_build torch_mlir "$python_version"
          ;;
        torch-mlir-core)
          clean_wheels torch_mlir_core "$python_version"
          build_torch_mlir_core
          run_audit_wheel torch_mlir_core "$python_version"
          clean_build torch_mlir_core "$python_version"
          ;;
        out-of-tree)
          setup_venv "$python_version" "$TM_TORCH_VERSION"
          build_out_of_tree "$TM_USE_PYTORCH_BINARY" "$python_version" "$TM_TORCH_VERSION"
          if [ "${TM_UPDATE_ODS_AND_ABSTRACT_INTERP_LIB}" == "ON" ]; then
            pushd /main_checkout/torch-mlir
            TORCH_MLIR_BUILD_DIR=/main_checkout/torch-mlir/build_oot ./build_tools/update_torch_ods.sh
            TORCH_MLIR_BUILD_DIR=/main_checkout/torch-mlir/build_oot ./build_tools/update_abstract_interp_lib.sh
            popd
          fi
          if [ "${TM_SKIP_TESTS}" == "OFF" ]; then
            test_out_of_tree
          fi
          ;;
        in-tree)
          setup_venv "$python_version" "$TM_TORCH_VERSION"
          build_in_tree "$TM_USE_PYTORCH_BINARY" "$python_version" "$TM_TORCH_VERSION"
          if [ "${TM_UPDATE_ODS_AND_ABSTRACT_INTERP_LIB}" == "ON" ]; then
            pushd /main_checkout/torch-mlir
            ./build_tools/update_torch_ods.sh
            ./build_tools/update_abstract_interp_lib.sh
            popd
          fi
          if [ "${TM_SKIP_TESTS}" == "OFF" ]; then
            test_in_tree "$TM_TORCH_VERSION";
          fi
          ;;
        *)
          echo "Unrecognized package '$package'"
          exit 1
          ;;
      esac
    done
  done
}


function build_in_tree() {
  local torch_from_bin="$1"
  local python_version="$2"

  local torch_version="$3"
  local enable_ltc="ON"
  if [[ "${torch_version}" == "stable" ]]
  then
    enable_ltc="OFF"
  fi

  echo ":::: Build in-tree Torch from binary: $torch_from_bin with Python: $python_version"
  cmake -GNinja -B/main_checkout/torch-mlir/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=clang \
      -DCMAKE_CXX_COMPILER=clang++ \
      -DCMAKE_EXE_LINKER_FLAGS_INIT="-fuse-ld=lld" \
      -DCMAKE_MODULE_LINKER_FLAGS_INIT="-fuse-ld=lld" \
      -DCMAKE_SHARED_LINKER_FLAGS_INIT="-fuse-ld=lld" \
      -DLLVM_ENABLE_ASSERTIONS=ON \
      -DCMAKE_C_COMPILER_LAUNCHER=ccache \
      -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
      -DLLVM_ENABLE_PROJECTS=mlir \
      -DLLVM_EXTERNAL_PROJECTS="torch-mlir" \
      -DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR="/main_checkout/torch-mlir" \
      -DLLVM_TARGETS_TO_BUILD=host \
      -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
      -DTORCH_MLIR_ENABLE_LTC=${enable_ltc} \
      -DTORCH_MLIR_USE_INSTALLED_PYTORCH="$torch_from_bin" \
      -DTORCH_MLIR_SRC_PYTORCH_REPO=${TORCH_MLIR_SRC_PYTORCH_REPO} \
      -DTORCH_MLIR_SRC_PYTORCH_BRANCH=${TORCH_MLIR_SRC_PYTORCH_BRANCH} \
      -DTM_PYTORCH_INSTALL_WITHOUT_REBUILD=${TM_PYTORCH_INSTALL_WITHOUT_REBUILD} \
      -DPython3_EXECUTABLE="$(which python3)" \
      /main_checkout/torch-mlir/externals/llvm-project/llvm
  cmake --build /main_checkout/torch-mlir/build --target tools/torch-mlir/all
  ccache -s
}

function _check_file_not_changed_by() {
  # _check_file_not_changed_by <cmd> <file>
  cmd="$1"
  file="$2"
  file_backup="$PWD/$(basename $file)"
  file_new="$PWD/$(basename $file).new"
  # Save the original file.
  cp "$file" "$file_backup"
  # Run the command to regenerate it.
  "$1" || return 1
  # Save the new generated file.
  cp "$file" "$file_new"
  # Restore the original file. We want this function to not change the user's
  # working tree state.
  mv "$file_backup" "$file"
  # We use git-diff as "just a diff program" (no SCM stuff) because it has
  # nicer output than regular `diff`.
  if ! git diff --no-index --quiet "$file" "$file_new"; then
    echo "#######################################################"
    echo "Generated file '${file}' is not up to date (see diff below)"
    echo ">>> Please run '${cmd}' to update it <<<"
    echo "#######################################################"
    git diff --no-index --color=always "$file" "$file_new"
    # TODO: Is there a better cleanup strategy that doesn't require duplicating
    # this inside and outside the `if`?
    rm "$file_new"
    return 1
  fi
  rm "$file_new"
}

function test_in_tree() {
  local torch_version="$1"
  
  echo ":::: Test in-tree"
  cmake --build /main_checkout/torch-mlir/build --target check-torch-mlir-all

  cd /main_checkout/torch-mlir/
  export PYTHONPATH="/main_checkout/torch-mlir/build/tools/torch-mlir/python_packages/torch_mlir:/main_checkout/torch-mlir/projects/pt1"

  case $torch_version in
    nightly)
      echo ":::: Test with nightly torch"

      echo ":::: Check that update_abstract_interp_lib.sh has been run"
      _check_file_not_changed_by ./build_tools/update_abstract_interp_lib.sh lib/Dialect/Torch/Transforms/AbstractInterpLibrary.cpp

      echo ":::: Check that update_torch_ods.sh has been run"
      _check_file_not_changed_by ./build_tools/update_torch_ods.sh include/torch-mlir/Dialect/Torch/IR/GeneratedTorchOps.td

      echo ":::: Run Lazy Tensor Core e2e integration tests"
      python -m e2e_testing.main --config=lazy_tensor_core -v

      echo ":::: Run Linalg e2e integration tests"
      python -m e2e_testing.main --config=linalg -v

      echo ":::: Run Onnx e2e integration tests"
      python -m e2e_testing.main --config=onnx -v

      # Dynamo is changing a lot in nightly versions, and thus the implementation
      # tends to become incompatible to the stable version.
      echo ":::: Run TorchDynamo e2e integration tests"
      python -m e2e_testing.main --config=torchdynamo -v
      ;;
    stable)
      echo ":::: Test with stable torch"

      # Disabled until the next stable PyTorch release (v2.1) is available
      # echo ":::: Run Lazy Tensor Core e2e integration tests in experimental mode"
      # python -m e2e_testing.main --config=lazy_tensor_core -v --ignore_failures
      ;;
    *)
      echo "Unrecognized torch version '$torch_version'"
      exit 1
      ;;
    esac

  echo ":::: Run make_fx + TOSA e2e integration tests"
  python -m e2e_testing.main --config=make_fx_tosa -v

  echo ":::: Run TOSA e2e integration tests"
  python -m e2e_testing.main --config=tosa -v
}

function setup_venv() {
  local python_version="$1"
  local torch_version="$2"

  echo ":::: Setting up VENV with Python: $python_version PyTorch $torch_version"
  python3 -m venv /main_checkout/torch-mlir/docker_venv
  source /main_checkout/torch-mlir/docker_venv/bin/activate

  echo ":::: pip installing dependencies"
  python3 -m pip install --no-cache-dir -r /main_checkout/torch-mlir/externals/llvm-project/mlir/python/requirements.txt

  case $torch_version in
    nightly)
      echo ":::: Using nightly dependencies"
      python3 -m pip install --no-cache-dir -r /main_checkout/torch-mlir/requirements.txt
      python3 -m pip install --no-cache-dir -r /main_checkout/torch-mlir/torchvision-requirements.txt
      ;;
    stable)
      echo ":::: Using stable dependencies"
      python3 -m pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
      python3 -m pip install --no-cache-dir -r /main_checkout/torch-mlir/build-requirements.txt
      ;;
    *)
      echo "Unrecognized torch version '$torch_version'"
      exit 1
      ;;
  esac

  python3 -m pip install --no-cache-dir -r /main_checkout/torch-mlir/test-requirements.txt
}

function build_out_of_tree() {
  local torch_from_bin="$1"
  local python_version="$2"
  local torch_version="$3"
  echo ":::: Build out-of-tree Torch from binary: $torch_from_bin with Python: $python_version ($torch_version)"

  local enable_ltc="ON"
  if [[ "${torch_version}" == "stable" ]]
  then
    enable_ltc="OFF"
  fi

  if [ ! -d "/main_checkout/torch-mlir/llvm-build/lib/cmake/mlir/" ]
  then
  echo ":::: LLVM / MLIR is not built so building it first.."
    cmake -GNinja -B/main_checkout/torch-mlir/llvm-build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=clang \
        -DCMAKE_CXX_COMPILER=clang++ \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_EXE_LINKER_FLAGS_INIT="-fuse-ld=lld" \
        -DCMAKE_MODULE_LINKER_FLAGS_INIT="-fuse-ld=lld" \
        -DCMAKE_SHARED_LINKER_FLAGS_INIT="-fuse-ld=lld" \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DLLVM_ENABLE_PROJECTS=mlir \
        -DLLVM_TARGETS_TO_BUILD=host \
        -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
        -DPython3_EXECUTABLE="$(which python3)" \
        /main_checkout/torch-mlir/externals/llvm-project/llvm
    cmake --build /main_checkout/torch-mlir/llvm-build
  fi

  # Incremental builds come here directly and can run cmake if required.
  cmake -GNinja -B/main_checkout/torch-mlir/build_oot \
      -DCMAKE_C_COMPILER=clang \
      -DCMAKE_CXX_COMPILER=clang++ \
      -DCMAKE_C_COMPILER_LAUNCHER=ccache \
      -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
      -DCMAKE_EXE_LINKER_FLAGS_INIT="-fuse-ld=lld" \
      -DCMAKE_MODULE_LINKER_FLAGS_INIT="-fuse-ld=lld" \
      -DCMAKE_SHARED_LINKER_FLAGS_INIT="-fuse-ld=lld" \
      -DLLVM_DIR="/main_checkout/torch-mlir/llvm-build/lib/cmake/llvm/" \
      -DMLIR_DIR="/main_checkout/torch-mlir/llvm-build/lib/cmake/mlir/" \
      -DMLIR_ENABLE_BINDINGS_PYTHON=OFF \
      -DTORCH_MLIR_ENABLE_LTC=${enable_ltc} \
      -DTORCH_MLIR_USE_INSTALLED_PYTORCH="$torch_from_bin" \
      -DTORCH_MLIR_SRC_PYTORCH_REPO=${TORCH_MLIR_SRC_PYTORCH_REPO} \
      -DTORCH_MLIR_SRC_PYTORCH_BRANCH=${TORCH_MLIR_SRC_PYTORCH_BRANCH} \
      -DTM_PYTORCH_INSTALL_WITHOUT_REBUILD=${TM_PYTORCH_INSTALL_WITHOUT_REBUILD} \
      -DPython3_EXECUTABLE="$(which python3)" \
      /main_checkout/torch-mlir
  cmake --build /main_checkout/torch-mlir/build_oot
  ccache -s
}

function test_out_of_tree() {
  echo ":::: Test out-of-tree"
  cmake --build /main_checkout/torch-mlir/build_oot --target check-torch-mlir-all
}

function clean_build() {
  # clean up for recursive runs
  local package="$1"
  local python_version="$2"
  echo ":::: Clean build dir $package $python_version"
  rm -rf /main_checkout/torch-mlir/build /main_checkout/torch-mlir/llvm-build /main_checkout/torch-mlir/docker_venv  /main_checkout/torch-mlir/libtorch
}

function build_torch_mlir() {
  local torch_version="$1"
  case $torch_version in
    nightly)
      echo ":::: Using nightly dependencies"
      python -m pip install --no-cache-dir -r /main_checkout/torch-mlir/requirements.txt \
        --extra-index-url https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
      CMAKE_GENERATOR=Ninja \
      TORCH_MLIR_PYTHON_PACKAGE_VERSION=${TORCH_MLIR_PYTHON_PACKAGE_VERSION} \
      python -m pip wheel -v -w /wheelhouse /main_checkout/torch-mlir \
        -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html \
        -r /main_checkout/torch-mlir/whl-requirements.txt
      ;;
    stable)
      echo ":::: Using stable dependencies"
      python3 -m pip install --no-cache-dir torch torchvision
      python3 -m pip install --no-cache-dir -r /main_checkout/torch-mlir/build-requirements.txt
      CMAKE_GENERATOR=Ninja \
      TORCH_MLIR_PYTHON_PACKAGE_VERSION=${TORCH_MLIR_PYTHON_PACKAGE_VERSION} \
      python -m pip wheel -v -w /wheelhouse /main_checkout/torch-mlir
      ;;
    *)
      echo "Unrecognized torch version '$torch_version'"
      exit 1
      ;;
  esac
}

function run_audit_wheel() {
  local wheel_basename="$1"
  local python_version="$2"
  generic_wheel="/wheelhouse/${wheel_basename}-${TORCH_MLIR_PYTHON_PACKAGE_VERSION}-${python_version}-linux_${arch}.whl"
  echo ":::: Auditwheel $generic_wheel"
  auditwheel repair -w /wheelhouse "$generic_wheel"
  rm "$generic_wheel"
}

function build_torch_mlir_core() {
  python -m pip install --no-cache-dir -r /main_checkout/torch-mlir/build-requirements.txt
  CMAKE_GENERATOR=Ninja \
  TORCH_MLIR_PYTHON_PACKAGE_VERSION=${TORCH_MLIR_PYTHON_PACKAGE_VERSION} \
  TORCH_MLIR_ENABLE_JIT_IR_IMPORTER=0 \
  TORCH_MLIR_ENABLE_ONLY_MLIR_PYTHON_BINDINGS=1 \
  python -m pip wheel -v -w /wheelhouse /main_checkout/torch-mlir
}

function clean_wheels() {
  local wheel_basename="$1"
  local python_version="$2"
  echo ":::: Clean wheels $wheel_basename $python_version"
  rm -f /wheelhouse/"${wheel_basename}"-*-"${python_version}"-*.whl
}

# Trampoline to the docker container if running on the host.
if [ -z "${__MANYLINUX_BUILD_WHEELS_IN_DOCKER-}" ]; then
  for package in $TM_PACKAGES; do
    echo "******************** BUILDING PACKAGE ${package} (host) *************"
    run_on_host "${package} $@"
  done
else
  run_in_docker "$@"
fi
