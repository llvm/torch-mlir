# Sharp edge: Torch extensions need to use the same pybind11 that torch
# was compiled with, or else there will be issues in cross module exception
# handling (which will abort instead of raise). We circumvent the possibility
# by forcing the torch directories first.
include_directories(BEFORE
  ${TORCH_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${Python3_INCLUDE_DIRS}
  )
link_directories("${TORCH_INSTALL_PREFIX}/lib")

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
target_link_libraries(TorchMLIRJITIRImporter
  TorchMLIRAggregateCAPI
  ${TORCH_LIBRARIES}
  )
message(STATUS "TORCH_CXXFLAGS=${TORCH_CXXFLAGS}")
set_target_properties(TorchMLIRJITIRImporter PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${TORCH_MLIR_PYTHON_PACKAGES_DIR}/torch_mlir/torch_mlir/_mlir_libs"
  OUTPUT_NAME lib_jit_ir_importer
  PREFIX ""
  SUFFIX ".a"
  CXX_VISIBILITY_PRESET "default"
  COMPILE_FLAGS "${TORCH_CXXFLAGS}"
  )

# Separate Pybind MODULE due to issues with a SHARED library.
# https://github.com/llvm/torch-mlir/issues/1154
add_library(TorchMLIRJITIRImporterPybind MODULE
  class_annotator_pybind.cpp
  get_registered_ops.cpp
  import_options_pybind.cpp
  init_python_bindings.cpp
  module_builder.cpp
  )
add_dependencies(TorchMLIRJITIRImporterPybind
  TorchMLIRJITIRImporter
  )
target_link_libraries(TorchMLIRJITIRImporterPybind
  ${TORCH_LIBRARIES}
  torch_python
  TorchMLIRJITIRImporter
  )

# On static Python builds, there may not be Python libraries to link against
# (they will late bind at runtime from the executable). We have to condition
# this because in that case it is set to NOTFOUND and CMake will consider
# this an error.
if(Python3_LIBRARIES)
  target_link_libraries(TorchMLIRJITIRImporterPybind
    ${Python3_LIBRARIES}
  )
endif()

message(STATUS "TORCH_CXXFLAGS=${TORCH_CXXFLAGS}")
set_target_properties(TorchMLIRJITIRImporterPybind PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${TORCH_MLIR_PYTHON_PACKAGES_DIR}/torch_mlir/torch_mlir/_mlir_libs"
  OUTPUT_NAME _jit_ir_importer
  PREFIX "${PYTHON_MODULE_PREFIX}"
  SUFFIX "${PYTHON_MODULE_EXTENSION}"
  CXX_VISIBILITY_PRESET "hidden"
  COMPILE_FLAGS "${TORCH_CXXFLAGS}"
  )
mlir_python_setup_extension_rpath(TorchMLIRJITIRImporterPybind)

torch_mlir_python_target_compile_options(TorchMLIRJITIRImporterPybind)
mlir_check_all_link_libraries(TorchMLIRJITIRImporterPybind)
