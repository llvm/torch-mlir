# Setup PyTorch
include(TorchMLIRPyTorch)
TorchMLIRProbeForPyTorchInstall()
find_package(Torch 1.8 REQUIRED)
TorchMLIRConfigurePyTorch()

# Python sources
declare_mlir_python_sources(TorchMLIRPythonSources.CustomOp
  ROOT_DIR "${TORCH_MLIR_PYTHON_ROOT_DIR}"
  ADD_TO_PARENT TorchMLIRPythonSources
  SOURCES_GLOB
    _torch_mlir_custom_op_example/__init__.py
)

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
