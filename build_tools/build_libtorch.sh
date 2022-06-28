#!/usr/bin/env bash

set -xe pipefail

SRC_ROOT="$( cd "$(dirname "$0")" ; pwd -P)/.."
PYTORCH_ROOT=${PYTORCH_ROOT:-$SRC_ROOT/externals/pytorch}
PYTORCH_INSTALL_PATH=${PYTORCH_INSTALL_PATH:-$SRC_ROOT/libtorch}
PYTORCH_BRANCH="${PYTORCH_BRANCH:-master}"
LIBTORCH_VARIANT="${LIBTORCH_VARIANT:-static-without-deps}"
PT_C_COMPILER="${PT_C_COMPILER:-clang}"
PT_CXX_COMPILER="${PT_CXX_COMPILER:-clang++}"

echo "SRC_ROOT=${SRC_ROOT}"
echo "PYTORCH_ROOT=${PYTORCH_ROOT}"
echo "PYTORCH_BRANCH=${PYTORCH_BRANCH}"
echo "LIBTORCH_VARIANT=${LIBTORCH_VARIANT}"

if [[ "$LIBTORCH_VARIANT" == *"cxx11-abi"* ]]; then
  echo _GLIBCXX_USE_CXX11_ABI=1
  export _GLIBCXX_USE_CXX11_ABI=1
  CXX_ABI=1
  LIBTORCH_ABI="cxx11-abi-"
else
  echo _GLIBCXX_USE_CXX11_ABI=0
  export _GLIBCXX_USE_CXX11_ABI=0
  CXX_ABI=0
  LIBTORCH_ABI=
fi

retry () {
  $* || (sleep 1 && $*) || (sleep 2 && $*) || (sleep 4 && $*) || (sleep 8 && $*)
}

install_requirements() {
  pip install -qr $PYTORCH_ROOT/requirements.txt
  pip list
}

# Check for an existing libtorch at $PYTORCH_ROOT
check_existing_libtorch() {
  if [[ -f "$PYTORCH_INSTALL_PATH/lib/libtorch.so" ]]; then
    echo "Existing PYTORCH shared build found.. skipping build"
    return 0
  elif [[ -f "$PYTORCH_INSTALL_PATH/lib/libtorch.a" ]]; then
    echo "Existing PYTORCH static build found.. skipping build"
    return 0
  elif [[ -f "$PYTORCH_INSTALL_PATH/lib/libtorch.dylib" ]]; then
    echo "Existing PYTORCH shared dylib found.. skipping build"
    return 0
  fi
  return 1
}

# Download and unzip into externals/pytorch/libtorch
MACOS_X86_URL="https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-latest.zip"
# Download here (Pre-cxx11 ABI): https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
# Download here (cxx11 ABI): https://download.pytorch.org/libtorch/nightly/cpu/libtorch-cxx11-abi-shared-with-deps-latest.zip
# Download static here (cxx11 ABI): https://download.pytorch.org/libtorch/nightly/cpu/libtorch-cxx11-abi-static-with-deps-latest.zip
# static builds are broken upstream and ship shared libraries anyway. Hopefully we can reland the fix upstream.
LINUX_X86_URL="https://download.pytorch.org/libtorch/nightly/cpu/libtorch-static-without-deps-latest.zip"

download_libtorch() {
  cd $SRC_ROOT
  if [[ $(uname -s) = 'Darwin' ]]; then
  echo "Apple macOS detected"
  if [[ $(uname -m) == 'arm64' ]]; then
    echo "${Red}Apple M1 Detected...no libtorch/ binaries available"
    return 1
  else
    echo "Apple x86_64 Detected"
    DOWNLOAD_URL=${MACOS_X86_URL}
  fi
elif [[ $(uname -s) = 'Linux' ]]; then
  echo "$Linux detected"
  DOWNLOAD_URL=${LINUX_X86_URL}
else
  echo "OS not detected. Pray and Play"
  return 1
fi
  curl -O ${DOWNLOAD_URL}
  unzip -o libtorch-*.zip
  if [[ -f "$PYTORCH_INSTALL_PATH/lib/libtorch.so" ]]; then
    echo "Verifying Pytorch install -- libtorch.so found"
    return 0
  elif [[ -f "$PYTORCH_INSTALL_PATH/lib/libtorch.a" ]]; then
    echo "Verifying Pytorch install -- libtorch.a found"
    return 0
  fi
  return 1
}

checkout_pytorch() {
  if [[ ! -d "$PYTORCH_ROOT" ]]; then
    git clone https://github.com/pytorch/pytorch $PYTORCH_ROOT
  fi
  cd $PYTORCH_ROOT
  git fetch --all
  git checkout ${PYTORCH_BRANCH}
  git submodule update --init --recursive
}

build_pytorch() {
  BUILD_SHARED_VAR="ON"
  if [[ $LIBTORCH_VARIANT = *"static"* ]]; then
    BUILD_SHARED_VAR="OFF"
  fi
  cd $PYTORCH_ROOT
  BUILD_SHARED_LIBS=${BUILD_SHARED_VAR} BUILD_TESTS=OFF USE_GLOO=OFF USE_PYTORCH_QNNPACK=OFF USE_OPENMP=OFF  USE_OBSERVERS=OFF USE_KINETO=OFF USE_EIGEN_FOR_BLAS=OFF _GLIBCXX_USE_CXX11_ABI=${CXX_ABI} USE_NCCL=OFF INTERN_DISABLE_ONNX=OFF BUILD_PYTHONLESS=1 USE_CUDA=OFF USE_MKL=OFF USE_XNNPACK=OFF USE_DISTRIBUTED=OFF USE_BREAKPAD=OFF USE_MKLDNN=OFF USE_QNNPACK=OFF USE_NNPACK=OFF ONNX_ML=OFF python setup.py build
}

package_pytorch() {
  mkdir -p libtorch/{lib,bin,include,share}

  # Copy over all lib files
  cp -rv build/lib/*                libtorch/lib/
  cp -rv build/lib*/torch/lib/*     libtorch/lib/

  # Copy over all include files
  cp -rv build/include/*            libtorch/include/
  cp -rv build/lib*/torch/include/* libtorch/include/

  # Copy over all of the cmake files
  cp -rv build/lib*/torch/share/*   libtorch/share/

  echo "${PYTORCH_BUILD_VERSION}" > libtorch/build-version
  echo "$(pushd $PYTORCH_ROOT && git rev-parse HEAD)" > libtorch/build-hash
  echo "Installing libtorch in ${PYTORCH_ROOT}/../../"
  echo "deleteing old ${PYTORCH_ROOT}/../../libtorch"
  rm -rf ${PYTORCH_ROOT}/../../libtorch
  mv libtorch ${PYTORCH_ROOT}/../../
}

#main
if check_existing_libtorch; then
  echo "Found libtorch"
  echo "Remove libtorch/ if you want to re-download or rebuild"
else
  if [ $SRC_BUILD ]; then
    echo "Building libtorch from source"
    checkout_pytorch
    install_requirements
    build_pytorch
    package_pytorch
  else
    echo "Downloading libtorch"
    download_libtorch
  fi
fi
