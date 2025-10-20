declare_mlir_python_sources(TorchMLIRE2ETestPythonSources)

declare_mlir_python_sources(TorchMLIRE2ETestPythonSources.Core
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  ADD_TO_PARENT TorchMLIRE2ETestPythonSources
  SOURCES_GLOB
    *.py
)

add_mlir_python_modules(TorchMLIRE2ETestPythonModules
  ROOT_PREFIX "${TORCH_MLIR_PYTHON_PACKAGES_DIR}/torch_mlir/torch_mlir_e2e_test"
  INSTALL_PREFIX "python_packages/torch_mlir/torch_mlir_e2e_test"
  DECLARED_SOURCES TorchMLIRE2ETestPythonSources
  )
