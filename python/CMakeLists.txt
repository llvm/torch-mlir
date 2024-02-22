# Disables generation of "version soname" (i.e. libFoo.so.<version>), which
# causes pure duplication as part of Python wheels.
set(CMAKE_PLATFORM_NO_VERSIONED_SONAME ON)

# The directory at which the Python import tree begins.
# See documentation for `declare_mlir_python_sources`'s ROOT_DIR
# argument.
set(TORCH_MLIR_PYTHON_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/torch_mlir")


# We vendor our own MLIR instance in the `torch_mlir` namespace.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=torch_mlir.")

################################################################################
# Sources
################################################################################

declare_mlir_python_sources(TorchMLIRPythonSources)
declare_mlir_python_sources(TorchMLIRPythonExtensions)

declare_mlir_python_sources(TorchMLIRPythonSources.Dialects
  ROOT_DIR "${TORCH_MLIR_PYTHON_ROOT_DIR}"
  ADD_TO_PARENT TorchMLIRPythonSources
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT TorchMLIRPythonSources.Dialects
  ROOT_DIR "${TORCH_MLIR_PYTHON_ROOT_DIR}"
  TD_FILE dialects/TorchBinding.td
  SOURCES dialects/torch/__init__.py
  DIALECT_NAME torch
)

declare_mlir_python_sources(TorchMLIRPythonSources.Importers
  ROOT_DIR "${TORCH_MLIR_PYTHON_ROOT_DIR}"
  ADD_TO_PARENT TorchMLIRPythonSources
  SOURCES
    extras/fx_importer.py
    extras/onnx_importer.py
)

declare_mlir_python_sources(TorchMLIRPythonSources.PublicAPI
  ROOT_DIR "${TORCH_MLIR_PYTHON_ROOT_DIR}"
  ADD_TO_PARENT TorchMLIRPythonSources
  SOURCES
    fx.py
    extras/fx_decomp_util.py
)

declare_mlir_python_sources(TorchMLIRPythonSources.Tools
  ROOT_DIR "${TORCH_MLIR_PYTHON_ROOT_DIR}"
  ADD_TO_PARENT TorchMLIRPythonSources
  SOURCES
    tools/import_onnx/__main__.py
)

declare_mlir_python_sources(TorchMLIRSiteInitialize
  ROOT_DIR "${TORCH_MLIR_PYTHON_ROOT_DIR}"
  ADD_TO_PARENT TorchMLIRPythonSources
  SOURCES
    _mlir_libs/_site_initialize_0.py
)

################################################################################
# Extensions
################################################################################

declare_mlir_python_extension(TorchMLIRPythonExtensions.Main
  MODULE_NAME _torchMlir
  ADD_TO_PARENT TorchMLIRPythonExtensions
  SOURCES
    TorchMLIRModule.cpp
  EMBED_CAPI_LINK_LIBS
    TorchMLIRCAPI
  PRIVATE_LINK_LIBS
    LLVMSupport
)

################################################################################
# Generate packages and shared library
# Downstreams typically will not use these, but they are useful for local
# testing.
################################################################################

set(_source_components
  # TODO: Core is now implicitly building/registering all dialects, increasing
  # build burden by ~5x. Make it stop.
  # TODO: Reduce dependencies. We need ExecutionEngine and a bunch of passes
  # for the reference backend, but logically they can be separate. But seemingly
  # the only way to handle that is to create a separate mlir python package
  # tree, which seems excessive.
  MLIRPythonSources
  MLIRPythonExtension.Core
  MLIRPythonExtension.RegisterEverything
  TorchMLIRPythonSources
  TorchMLIRPythonExtensions
  TorchMLIRSiteInitialize

  # Sources related to optional Torch extension dependent features. Typically
  # empty unless if project features are enabled.
  TorchMLIRPythonTorchExtensionsSources
)

add_mlir_python_common_capi_library(TorchMLIRAggregateCAPI
  INSTALL_COMPONENT TorchMLIRPythonModules
  INSTALL_DESTINATION python_packages/torch_mlir/torch_mlir/_mlir_libs
  OUTPUT_DIRECTORY "${TORCH_MLIR_PYTHON_PACKAGES_DIR}/torch_mlir/torch_mlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT ".."
  DECLARED_SOURCES ${_source_components}
)

add_mlir_python_modules(TorchMLIRPythonModules
  ROOT_PREFIX "${TORCH_MLIR_PYTHON_PACKAGES_DIR}/torch_mlir/torch_mlir"
  INSTALL_PREFIX "python_packages/torch_mlir/torch_mlir"
  DECLARED_SOURCES ${_source_components}
  COMMON_CAPI_LINK_LIBS
    TorchMLIRAggregateCAPI
  )
