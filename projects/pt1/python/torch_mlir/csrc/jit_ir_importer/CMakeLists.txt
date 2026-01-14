# Separate Pybind MODULE due to issues with a SHARED library.
# https://github.com/llvm/torch-mlir/issues/1154
Python3_add_library(TorchMLIRJITIRImporterPybind MODULE WITH_SOABI
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
  PRIVATE
    ${TORCH_LIBRARIES}
    torch_python
    TorchMLIRJITIRImporter
  )

# Prefer the imported target when available (handles static Python correctly).
if(TARGET Python3::Module)
  target_link_libraries(TorchMLIRJITIRImporterPybind PRIVATE Python3::Module)
elseif(Python3_LIBRARIES)
  # On some setups the imported target is unavailable; fall back to raw libs.
  target_link_libraries(TorchMLIRJITIRImporterPybind PRIVATE ${Python3_LIBRARIES})
endif()

set_target_properties(TorchMLIRJITIRImporterPybind PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${TORCH_MLIR_PYTHON_PACKAGES_DIR}/torch_mlir/torch_mlir/_mlir_libs"
  OUTPUT_NAME _jit_ir_importer
  CXX_VISIBILITY_PRESET "hidden"
  COMPILE_FLAGS "${TORCH_CXXFLAGS}"
  )
mlir_python_setup_extension_rpath(TorchMLIRJITIRImporterPybind)

torch_mlir_python_target_compile_options(TorchMLIRJITIRImporterPybind)
mlir_check_all_link_libraries(TorchMLIRJITIRImporterPybind)
