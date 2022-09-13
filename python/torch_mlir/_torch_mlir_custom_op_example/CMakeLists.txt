###########################################################################
# Setup PyTorch
###########################################################################

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/python/torch_mlir/cmake/modules")
include(TorchMLIRPyTorch)

TorchMLIRProbeForPyTorchInstall()
set(Torch_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../libtorch/share/cmake/Torch")

find_package(Torch 1.11 REQUIRED)

###########################################################################
# Library definition
###########################################################################

# C++ extension
include_directories(BEFORE
  ${TORCH_INCLUDE_DIRS}
)
add_library(torch_mlir_custom_op_example SHARED torch_mlir_custom_op_example.cpp)
target_link_libraries(torch_mlir_custom_op_example
  ${TORCH_LIBRARIES}
)
# Because the custom op library is a bit odd, we'd like it to stay with the
# Python component in the build directory.
set_target_properties(torch_mlir_custom_op_example PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${TORCH_MLIR_PYTHON_PACKAGES_DIR}/torch_mlir/torch_mlir/_torch_mlir_custom_op_example/"
  COMPILE_FLAGS "${TORCH_CXXFLAGS}"
)
torch_mlir_python_target_compile_options(torch_mlir_custom_op_example)
mlir_check_all_link_libraries(torch_mlir_custom_op_example)

