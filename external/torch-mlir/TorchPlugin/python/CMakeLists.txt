
## Declare the sources of the Python module.

declare_mlir_python_sources(TorchMLIRPluginPythonSources)

declare_mlir_python_sources(TorchMLIRPluginPythonSources.Core
  ADD_TO_PARENT TorchMLIRPluginPythonSources
  SOURCES_GLOB
    torch_mlir/*.py
    torch_mlir_utils/codegen/*.py
)

set(_source_components
  TorchMLIRPluginPythonSources
)

add_mlir_python_modules(TorchMLIRPluginPythonModules
  ROOT_PREFIX "${TORCH_MLIR_PYTHON_PACKAGES_DIR}/torch_mlir"
  INSTALL_PREFIX "python_packages/torch_mlir"
  DECLARED_SOURCES ${_source_components}
  COMMON_CAPI_LINK_LIBS
    TorchMLIRAggregateCAPI
  )
add_dependencies(TorchMLIRPluginPythonModules TorchMLIRTorchPlugin)
