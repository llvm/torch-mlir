# in-tree build
# FIXME: set MLIR_PDLL_TABLEGEN_EXE since mlir-hlo doesn't
cmake -GNinja -Bbuild \
  -DCMAKE_BUILD_TYPE=DEBUG \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DPython3_FIND_VIRTUALENV=ONLY \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_EXTERNAL_PROJECTS="torch-mlir;torch-mlir-dialects;mlir_hlo" \
  -DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR=`pwd` \
  -DLLVM_EXTERNAL_TORCH_MLIR_DIALECTS_SOURCE_DIR=`pwd`/externals/llvm-external-projects/torch-mlir-dialects \
  -DLLVM_EXTERNAL_MLIR_HLO_SOURCE_DIR=`pwd`/externals/mlir-hlo \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DLLVM_TARGETS_TO_BUILD=host \
  -DMLIR_PDLL_TABLEGEN_EXE=mlir-pdll \
  -DLIBTORCH_CACHE=ON \
  externals/llvm-project/llvm

# Additional quality of life CMake flags:
# Enable ccache:
#  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
# Enable LLD (links in seconds compared to minutes)
# -DCMAKE_EXE_LINKER_FLAGS_INIT="-fuse-ld=lld" -DCMAKE_MODULE_LINKER_FLAGS_INIT="-fuse-ld=lld" -DCMAKE_SHARED_LINKER_FLAGS_INIT="-fuse-ld=lld"
# Use --ld-path= instead of -fuse-ld=lld for clang > 13

# Build just torch-mlir (not all of LLVM)
# cmake --build build --target tools/torch-mlir/all
# 
# Run unit tests.
cmake --build build --target check-torch-mlir

# Build everything (including LLVM)
# cmake --build build
