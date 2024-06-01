#!/bin/bash

# Function to measure build time
measure_build_time() {
  local linker=$1
  local build_dir=$2
  local max_cores=$3
  local results_file="$results_dir/${linker}_${max_cores}_cores.txt"
  
  # Remove the build directory
  rm -rf "$build_dir"
  
  # Create build directory
  mkdir -p "$build_dir"
  
  echo "Configuring with $tool..."
  cmake -S /home/azureuser/buildbench/torch-mlir -B "$build_dir" \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_EXTERNAL_PROJECTS="torch-mlir" \
    -DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR="/home/azureuser/buildbench/torch-mlir" \
    -DCMAKE_BUILD_TYPE=Debug \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_EXTERNAL_PROJECTS=torch-mlir;torch-mlir-dialects \
    -DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR=${workspaceFolder} \
    -DLLVM_EXTERNAL_TORCH_MLIR_DIALECTS_SOURCE_DIR=${workspaceFolder}/externals/llvm-external-projects/torch-mlir-dialects \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_TARGETS_TO_BUILD=host \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_EXE_LINKER_FLAGS_INIT="$flags" \
    -DCMAKE_MODULE_LINKER_FLAGS_INIT="$flags" \
    -DCMAKE_SHARED_LINKER_FLAGS_INIT="$flags" \
    -DLIBTORCH_CACHE=ON \
    -DLIBTORCH_SRC_BUILD=ON \
    -DLIBTORCH_VARIANT=shared \
    -G Ninja
  
  # Verify the linker being used by introducing an intentional linker error
  echo "int main() { undefined_function(); return 0; }" > "$build_dir/linker_test.c"
  if ! cmake --build "$build_dir" --config Debug --target linker_test 2>&1 | grep -q "$linker"; then
    echo "Error: Linker mismatch. Expected $linker, but a different linker was used."
    exit 1
  fi
  rm "$build_dir/linker_test.c"
  
  local start_time=$(date +%s)
  cmake --build $build_dir --config Debug -- -j $max_cores
  local end_time=$(date +%s)
  local build_time=$((end_time - start_time))
  echo "Build time with $linker ($max_cores cores): $build_time seconds" | tee -a "$results_file"
  
  # Perform an incremental build and measure the link time
  echo "Performing incremental build..."
  touch "$build_dir/src/incremental_change.c"
  incremental_start_time=$(date +%s)
  cmake --build "$build_dir" --config Debug -- -j "$max_cores"
  incremental_end_time=$(date +%s)
  incremental_build_time=$((incremental_end_time - incremental_start_time))
  echo "Incremental build time with $linker ($max_cores cores): $incremental_build_time seconds" | tee -a "$results_file"
}

# Create results directory
results_dir="/home/azureuser/buildbench/torch-mlir/benchmark_results"
mkdir -p "$results_dir"

# Define build tools and their corresponding flags
build_tools=(
  "LLD   --ld-path=lld"
  "MOLD  --ld-path=mold"
  "LD    "
)

# Iterate through build tools
for tool_entry in "${build_tools[@]}"; do
  IFS=' ' read -r tool flags <<< "$tool_entry"
  build_dir="/home/azureuser/buildbench/torch-mlir/build-$tool"
  
  # Benchmark build with the current tool
  for cores in 1 2 4 8 16; do
    measure_build_time "$tool" "$build_dir" $cores
  done
done
