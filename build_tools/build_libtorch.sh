#!/usr/bin/env bash

set -xeu -o pipefail

SRC_ROOT="$( cd "$(dirname "$0")" ; pwd -P)/.."
PYTORCH_ROOT=${PYTORCH_ROOT:-$SRC_ROOT/externals/pytorch}
PYTORCH_INSTALL_PATH=${PYTORCH_INSTALL_PATH:-$SRC_ROOT/libtorch}
PYTORCH_REPO="${PYTORCH_REPO:-pytorch/pytorch}"
PYTORCH_BRANCH="${PYTORCH_BRANCH:-master}"
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
echo "PYTORCH_REPO=${PYTORCH_REPO}"
echo "PYTORCH_BRANCH=${PYTORCH_BRANCH}"
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
    git clone --depth 1 --single-branch --branch "${PYTORCH_BRANCH}" https://github.com/"$PYTORCH_REPO" "$PYTORCH_ROOT"
  fi
  cd "$PYTORCH_ROOT"
  git reset --hard HEAD
  git clean -df
  git submodule update --init --depth 1 --recursive
}

build_pytorch() {
  sed -i.bak 's/INTERN_USE_EIGEN_BLAS ON/INTERN_USE_EIGEN_BLAS OFF/g' "${PYTORCH_ROOT}"/CMakeLists.txt
  sed -i.bak 's/set(BUILD_PYTHON OFF)/set(BUILD_PYTHON ON)/g' "${PYTORCH_ROOT}"/CMakeLists.txt
  sed -i.bak 's/set(INTERN_DISABLE_ONNX ON)/set(INTERN_DISABLE_ONNX OFF)/g' "${PYTORCH_ROOT}"/CMakeLists.txt
#  # around line 150 (under if(BUILD_LITE_INTERPRETER))
#  sed -i.bak 's/set(all_cpu_cpp ${generated_sources} ${core_generated_sources} ${cpu_kernel_cpp})/set(all_cpu_cpp "${${CMAKE_PROJECT_NAME}_SOURCE_DIR}\/build\/aten\/src\/ATen\/RegisterSchema.cpp")/g' "${PYTORCH_ROOT}"/aten/src/ATen/CMakeLists.txt
  sed -i.bak 's/set(all_cpu_cpp ${generated_sources} ${core_generated_sources} ${cpu_kernel_cpp})/set(all_cpu_cpp ${generated_sources} ${core_generated_sources})/g' "${PYTORCH_ROOT}"/aten/src/ATen/CMakeLists.txt
  sed -i.bak 's/append_filelist("aten_native_source_non_codegen_list" all_cpu_cpp)/append_filelist("core_sources_full" all_cpu_cpp)/g' "${PYTORCH_ROOT}"/aten/src/ATen/CMakeLists.txt
  sed -i.bak 's/PyTorchBackendDebugInfo/PyTorchBackendDebugInfoDummy/g' "${PYTORCH_ROOT}"/torch/csrc/jit/backends/backend_detail.cpp
  sed -i.bak 's/backend_debug_info->setDebugInfoMap(std::move(debug_info_map));/\/\/backend_debug_info->setDebugInfoMap(std::move(debug_info_map));/g' "${PYTORCH_ROOT}"/torch/csrc/jit/backends/backend_detail.cpp

  CMAKE_ARGS=()
  if [ -x "$(command -v ninja)" ]; then
    CMAKE_ARGS+=("-GNinja")
  fi

  CMAKE_ARGS+=("-DCMAKE_PREFIX_PATH=${PYTHON_BIN}")
  CMAKE_ARGS+=("-DPYTHON_EXECUTABLE=$($PYTHON_BIN -c 'import sys; print(sys.executable)')")

  # Necessary flags to hit minimal path
  export BUILD_PYTORCH_MOBILE_WITH_HOST_TOOLCHAIN=1
  CMAKE_ARGS+=("-DBUILD_LITE_INTERPRETER:BOOL=TRUE")
  CMAKE_ARGS+=("-DBUILD_SHARED_LIBS:BOOL=FALSE")
  # torch/csrc/jit/mobile/profiler_edge.cpp includes KinetoEdgeCPUProfiler::~KinetoEdgeCPUProfiler
  CMAKE_ARGS+=("-DUSE_KINETO:BOOL=TRUE")
  CMAKE_ARGS+=("-DUSE_LIGHTWEIGHT_DISPATCH:BOOL=TRUE")
  CMAKE_ARGS+=("-DSTATIC_DISPATCH_BACKEND=CPU")

#  CMAKE_ARGS+=("-DCMAKE_CXX_FLAGS=-Wl,--unresolved-symbols=ignore-all")
#  CMAKE_ARGS+=("-DCMAKE_EXE_LINKER_FLAGS=-Wl,--unresolved-symbols=ignore-all")
#  CMAKE_ARGS+=("-DCMAKE_SHARED_LINKER_FLAGS=-Wl,--unresolved-symbols=ignore-all")

  # Disable unused dependencies
  CMAKE_ARGS+=("-DBUILD_CUSTOM_PROTOBUF:BOOL=FALSE")
  CMAKE_ARGS+=("-DBUILD_TEST:BOOL=FALSE")
  CMAKE_ARGS+=("-DUSE_ASAN:BOOL=FALSE")
  CMAKE_ARGS+=("-DUSE_BLAS:BOOL=FALSE")
  CMAKE_ARGS+=("-DUSE_BREAKPAD:BOOL=FALSE")
  CMAKE_ARGS+=("-DUSE_CUDA:BOOL=FALSE")
  CMAKE_ARGS+=("-DUSE_CUDNN:BOOL=FALSE")
  CMAKE_ARGS+=("-DUSE_DISTRIBUTED:BOOL=FALSE")
  CMAKE_ARGS+=("-DUSE_FBGEMM:BOOL=FALSE")
  CMAKE_ARGS+=("-DUSE_GFLAGS=OFF")
  CMAKE_ARGS+=("-DUSE_GLOO:BOOL=FALSE")
  CMAKE_ARGS+=("-DUSE_LAPACK:BOOL=FALSE")
  CMAKE_ARGS+=("-DUSE_LEVELDB=OFF")
  CMAKE_ARGS+=("-DUSE_LMDB=OFF")
  CMAKE_ARGS+=("-DUSE_MKLDNN:BOOL=FALSE")
  CMAKE_ARGS+=("-DUSE_MPI=OFF")
  CMAKE_ARGS+=("-DUSE_NCCL:BOOL=FALSE")
  CMAKE_ARGS+=("-DUSE_NNPACK:BOOL=FALSE")
  CMAKE_ARGS+=("-DUSE_NUMPY:BOOL=FALSE")
  CMAKE_ARGS+=("-DUSE_OBSERVERS:BOOL=FALSE")
  CMAKE_ARGS+=("-DUSE_OPENCV=OFF")
  CMAKE_ARGS+=("-DUSE_OPENMP:BOOL=FALSE")
  CMAKE_ARGS+=("-DUSE_OPENMP=OFF")
  CMAKE_ARGS+=("-DUSE_PYTORCH_QNNPACK:BOOL=FALSE")
  CMAKE_ARGS+=("-DUSE_QNNPACK:BOOL=FALSE")
  CMAKE_ARGS+=("-DUSE_ROCM=OFF")
  CMAKE_ARGS+=("-DUSE_TENSORPIPE:BOOL=FALSE")
  CMAKE_ARGS+=("-DUSE_VALGRIND:BOOL=FALSE")
  CMAKE_ARGS+=("-DUSE_XNNPACK:BOOL=FALSE")

  CMAKE_ARGS+=("-DCMAKE_C_COMPILER=$(which clang)")
  CMAKE_ARGS+=("-DCMAKE_CXX_COMPILER=$(which clang++)")

#  CMAKE_ARGS+=("-DCMAKE_INSTALL_RPATH=${PYTORCH_INSTALL_PATH}/lib")


  BUILD_ROOT=${BUILD_ROOT:-"$PYTORCH_ROOT/build"}
  mkdir -p $BUILD_ROOT
  cd $BUILD_ROOT
  cmake --build . --target clean || echo "No build to clean."

  cmake "$PYTORCH_ROOT" \
      -DCMAKE_INSTALL_PREFIX=$PYTORCH_INSTALL_PATH \
      -DCMAKE_BUILD_TYPE=Release \
      "${CMAKE_ARGS[@]}"

  if [ "$(uname)" == 'Darwin' ]; then
    MAX_JOBS=$(sysctl -n hw.ncpu)
  else
    MAX_JOBS=$(nproc)
  fi

  cmake --build . --target install -- "-j${MAX_JOBS}"
}

package_pytorch() {
  mkdir -p libtorch
  if [[ -d "libtorch/lib" ]]; then
    rm -rf libtorch/{lib,bin,include,share}
  fi

  # Copy over all of the cmake files
  mv "${PYTORCH_ROOT}"/build_minimal/install/share     libtorch/
  # Copy over all lib files
  mv "${PYTORCH_ROOT}"/build_minimal/install/lib       libtorch/lib
  # Copy over all include files
  mv "${PYTORCH_ROOT}"/build_minimal/install/include   libtorch/include

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
