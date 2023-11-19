# Static library with core functionality.
# We can't use a shared library here, due to issues with linking on macOS-arm64 (the library itself won't build)
# For details, see: https://github.com/llvm/torch-mlir/runs/7919012376
add_library(TorchMLIRJITIRImporter STATIC
  class_annotator.cpp
  function_importer.cpp
  node_importer.cpp
  ivalue_importer.cpp
  torch_to_mlir_utils.cpp
  )
message(STATUS "Linking TorchMLIRJITImporter with ${TORCH_LIBRARIES}")
target_link_libraries(TorchMLIRJITIRImporter
  TorchMLIRAggregateCAPI
  ${TORCH_LIBRARIES}
  )
# Includes are relative to the csrc dir (i.e. #include "jit_ir_importer/...")
target_include_directories(TorchMLIRJITIRImporter PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/..
)
set_target_properties(TorchMLIRJITIRImporter PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${TORCH_MLIR_PYTHON_PACKAGES_DIR}/torch_mlir/torch_mlir/_mlir_libs"
  OUTPUT_NAME lib_jit_ir_importer
  PREFIX ""
  SUFFIX ".a"
  CXX_VISIBILITY_PRESET "default"
  COMPILE_FLAGS "${TORCH_CXXFLAGS}"
  )
