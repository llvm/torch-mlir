#!/usr/bin/env bash

set -xeu -o pipefail

SRC_ROOT="$( cd "$(dirname "$0")" ; pwd -P)/.."
PYTORCH_ROOT=${PYTORCH_ROOT:-$SRC_ROOT/externals/pytorch}
PYTORCH_INSTALL_PATH=${PYTORCH_INSTALL_PATH:-$SRC_ROOT/libtorch}
TORCH_MLIR_SRC_PYTORCH_REPO="${TORCH_MLIR_SRC_PYTORCH_REPO:-pytorch/pytorch}"
TORCH_MLIR_SRC_PYTORCH_BRANCH="${TORCH_MLIR_SRC_PYTORCH_BRANCH:-master}"
TM_PYTORCH_INSTALL_WITHOUT_REBUILD="${TM_PYTORCH_INSTALL_WITHOUT_REBUILD:-false}"
PT_C_COMPILER="${PT_C_COMPILER:-clang}"
PT_CXX_COMPILER="${PT_CXX_COMPILER:-clang++}"
CMAKE_OSX_ARCHITECTURES="${CMAKE_OSX_ARCHITECTURES:-x86_64}"
MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-12.0}"
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
echo "TORCH_MLIR_SRC_PYTORCH_REPO=${TORCH_MLIR_SRC_PYTORCH_REPO}"
echo "TORCH_MLIR_SRC_PYTORCH_BRANCH=${TORCH_MLIR_SRC_PYTORCH_BRANCH}"
echo "MACOSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET}"
echo "CMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}"

export CMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}
export MACOSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET}
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
    # ${TORCH_MLIR_SRC_PYTORCH_BRANCH} could be a branch name or a commit hash.
    # Although `git clone` can accept a branch name, the same command does not
    # accept a commit hash, so we instead use `git fetch`.  The alternative is
    # to clone the entire repository and then `git checkout` the requested
    # branch or commit hash, but that's too expensive.
    mkdir "${PYTORCH_ROOT}"
    cd "${PYTORCH_ROOT}"
    git init
    git remote add origin "https://github.com/${TORCH_MLIR_SRC_PYTORCH_REPO}"
    git fetch --depth=1 origin "${TORCH_MLIR_SRC_PYTORCH_BRANCH}"
    git reset --hard FETCH_HEAD
  else
    cd "${PYTORCH_ROOT}"
    git fetch --depth=1 origin "${TORCH_MLIR_SRC_PYTORCH_BRANCH}"
    git reset --hard FETCH_HEAD
  fi
  git clean -df
  git submodule update --init --depth 1 --recursive
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

  BUILD_CUSTOM_PROTOBUF=OFF \
  BUILD_LAZY_TS_BACKEND=ON \
  BUILD_LITE_INTERPRETER=0 \
  SELECTED_OP_LIST="$SRC_ROOT/build_tools/lightweight_dispatch_ops.yaml" \
  STATIC_DISPATCH_BACKEND="CPU" \
  USE_LIGHTWEIGHT_DISPATCH=1 \
  BUILD_SHARED_LIBS=ON \
  BUILD_CAFFE2_OPS=OFF \
  INTERN_BUILD_ATEN_OPS=OFF \
  ATEN_NO_TEST=OFF \
  USE_LITE_INTERPRETER_PROFILER=OFF \
  BUILD_TEST=OFF \
  GLIBCXX_USE_CXX11_ABI=1 \
  CMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES} \
  MACOSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET} \
  INTERN_BUILD_ATEN_OPS=OFF \
  INTERN_DISABLE_ONNX=ON \
  INTERN_USE_EIGEN_BLAS=ON \
  MAX_JOBS=${MAX_JOBS} \
  ONNX_ML=OFF \
  USE_BREAKPAD=OFF \
  USE_CUDA=OFF \
  USE_ITT=OFF \
  USE_DISTRIBUTED=OFF \
  USE_EIGEN_FOR_BLAS=OFF \
  USE_FBGEMM=OFF \
  USE_GLOO=OFF \
  USE_KINETO=ON \
  USE_MKL=OFF \
  USE_MKLDNN=OFF \
  USE_MPS=OFF \
  USE_NCCL=OFF \
  USE_NNPACK=OFF \
  USE_OBSERVERS=OFF \
  USE_OPENMP=OFF \
  USE_PYTORCH_QNNPACK=ON \
  USE_QNNPACK=OFF \
  USE_XNNPACK=OFF \
  USE_PRECOMPILED_HEADERS=1 \
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

unpack_pytorch() {
  PYTHON_SITE=`${PYTHON_BIN} -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])'`
  echo "wheel unpacking Pytorch..into ${PYTHON_SITE}"
  wheel unpack -d "$WHEELHOUSE"/unpack_tmp "$WHEELHOUSE"/*.whl
  mv "$WHEELHOUSE"/unpack_tmp/* "$PYTHON_SITE"/
}

#main
echo "Building libtorch from source"
wheel_exists=true
compgen -G "$WHEELHOUSE/*.whl" > /dev/null || wheel_exists=false
if [[ $TM_PYTORCH_INSTALL_WITHOUT_REBUILD != "true" || ${wheel_exists} == "false" ]]; then
  checkout_pytorch
  install_requirements
  build_pytorch
  package_pytorch
fi
if [[ $CMAKE_OSX_ARCHITECTURES = "arm64" ]]; then
  echo "${Yellow} Cross compiling for arm64 so unpacking PyTorch wheel for libs${NC}"
  unpack_pytorch
else
  echo "${Green} Installing the built PyTorch wheel ${NC}"
  install_pytorch
fi
