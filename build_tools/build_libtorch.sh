#!/usr/bin/env bash

set -xeu -o pipefail

SRC_ROOT="$( cd "$(dirname "$0")" ; pwd -P)/.."
PYTORCH_ROOT=${PYTORCH_ROOT:-$SRC_ROOT/externals/pytorch}
PYTORCH_INSTALL_PATH=${PYTORCH_INSTALL_PATH:-$SRC_ROOT/libtorch}
PYTORCH_REPO="${PYTORCH_REPO:-pytorch/pytorch}"
PYTORCH_BRANCH="${PYTORCH_BRANCH:-master}"
PT_C_COMPILER="${PT_C_COMPILER:-clang}"
PT_CXX_COMPILER="${PT_CXX_COMPILER:-clang++}"
CMAKE_OSX_ARCHITECTURES="${CMAKE_OSX_ARCHITECTURES:-arm64;x86_64}"
WHEELHOUSE="${WHEELHOUSE:-$SRC_ROOT/build_tools/python_deploy/wheelhouse}"
PYTHON_BIN="${TORCH_MLIR_PYTHON_VERSION:-python3}"
PIP_BIN="${TORCH_MLIR_PIP_VERSION:-pip3}"
CMAKE_C_COMPILER_LAUNCHER="${CMAKE_C_COMPILER_LAUNCHER:-""}"
CMAKE_CXX_COMPILER_LAUNCHER="${CMAKE_CXX_COMPILER_LAUNCHER:-""}"

Red='\033[0;31m'
Green='\033[0;32m'
Yellow='\033[1;33m'
NC='\033[0m'

echo "SRC_ROOT=${SRC_ROOT}"
echo "PYTORCH_ROOT=${PYTORCH_ROOT}"
echo "PYTORCH_REPO=${PYTORCH_REPO}"
echo "PYTORCH_BRANCH=${PYTORCH_BRANCH}"
echo "CMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}"
export CMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}
export CMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
export CMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}

retry () {
  "$@" || (sleep 1 && "$@") || (sleep 2 && "$@") || (sleep 4 && "$@") || (sleep 8 && "$@")
}

install_requirements() {
  ${PIP_BIN} install -qr $PYTORCH_ROOT/requirements.txt
  ${PIP_BIN} list
}

checkout_pytorch() {
  if [[ ! -d "$PYTORCH_ROOT" ]]; then
    git clone --depth 1 --single-branch --branch "${PYTORCH_BRANCH}" https://github.com/"$PYTORCH_REPO" "$PYTORCH_ROOT"
  fi
  cd "$PYTORCH_ROOT"
  git reset --hard HEAD
  git clean -df
  for dep in protobuf pocketfft cpuinfo FP16 psimd fmt sleef pybind11 onnx flatbuffers foxi; do
    git submodule update --init --depth 1 -- third_party/$dep
  done
  # setup.py will try to re-fetch
  sed -i.bak -E 's/^[[:space:]]+check_submodules()/#check_submodules()/g' setup.py
}

build_pytorch() {
  cd "$PYTORCH_ROOT"
  # Uncomment the next line if you want to iterate on source builds
  # ${PYTHON_BIN} setup.py clean
  rm -rf "${WHEELHOUSE:?}"/*

  if [[ -z "${MAX_JOBS:-""}" ]]; then
    if [[ "$(uname)" == 'Darwin' ]]; then
      MAX_JOBS=$(sysctl -n hw.ncpu)
    else
      MAX_JOBS=$(nproc)
    fi
  fi

  BUILD_SHARED_LIBS=ON \
  BUILD_TEST=OFF \
  GLIBCXX_USE_CXX11_ABI=1 \
  CMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES} \
  INTERN_BUILD_ATEN_OPS=OFF \
  INTERN_DISABLE_ONNX=ON \
  INTERN_USE_EIGEN_BLAS=ON \
  MAX_JOBS=${MAX_JOBS} \
  ONNX_ML=OFF \
  USE_BREAKPAD=OFF \
  USE_CUDA=OFF \
  USE_DISTRIBUTED=OFF \
  USE_EIGEN_FOR_BLAS=OFF \
  USE_FBGEMM=OFF \
  USE_GLOO=OFF \
  USE_KINETO=OFF \
  USE_MKL=OFF \
  USE_MKLDNN=OFF \
  USE_MPS=OFF \
  USE_NCCL=OFF \
  USE_NNPACK=OFF \
  USE_OBSERVERS=OFF \
  USE_OPENMP=OFF \
  USE_PYTORCH_QNNPACK=OFF \
  USE_QNNPACK=OFF \
  USE_XNNPACK=OFF \
  ${PYTHON_BIN} setup.py  bdist_wheel -d "$WHEELHOUSE"
}

package_pytorch() {
  mkdir -p libtorch
  if [[ -d "libtorch/lib" ]]; then
    rm -rf libtorch/{lib,bin,include,share}
  fi

  # Copy over all of the cmake files
  mv build/lib*/torch/share     libtorch/
  mv build/lib*/torch/include   libtorch/
  mv build/lib*/torch/lib       libtorch/
  # Copy over all lib files
  mv build/lib/*                libtorch/lib/
  # Copy over all include files
  mv build/include/*            libtorch/include/

  (pushd "$PYTORCH_ROOT" && git rev-parse HEAD) > libtorch/build-hash
  echo "Installing libtorch in ${PYTORCH_ROOT}/../../"
  echo "deleting old ${PYTORCH_ROOT}/../../libtorch"
  rm -rf "${PYTORCH_ROOT}"/../../libtorch
  mv libtorch "${PYTORCH_ROOT}"/../../
}

install_pytorch() {
  echo "pip installing Pytorch.."
  ${PIP_BIN} install  --force-reinstall $WHEELHOUSE/*
}

#main
echo "Building libtorch from source"
checkout_pytorch
install_requirements
build_pytorch
package_pytorch
install_pytorch
