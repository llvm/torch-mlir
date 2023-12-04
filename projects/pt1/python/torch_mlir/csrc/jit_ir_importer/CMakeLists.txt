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
