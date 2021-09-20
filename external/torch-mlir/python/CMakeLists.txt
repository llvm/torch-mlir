include(AddMLIRPython)

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

################################################################################
# Extensions
################################################################################

declare_mlir_python_extension(TorchMLIRPythonExtensions.Main
  MODULE_NAME _torchMlir
  ADD_TO_PARENT TorchMLIRPythonExtensions
  SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/TorchMLIRModule.cpp
  EMBED_CAPI_LINK_LIBS
    TorchMLIRCAPI
  PRIVATE_LINK_LIBS
    LLVMSupport
)

################################################################################
# Optionally handle JIT IR importer.
################################################################################

if(TORCH_MLIR_ENABLE_JIT_IR_IMPORTER)
  add_subdirectory(torch_mlir/dialects/torch/importer/jit_ir)
endif()

################################################################################
# Generate packages and shared library
# Downstreams typically will not use these, but they are useful for local
# testing.
################################################################################

set(_source_components
  # TODO: Core is now implicitly building/registering all dialects, increasing
  # build burden by ~5x. Make it stop.
  MLIRPythonSources.Core
  MLIRPythonSources.Dialects.builtin
  MLIRPythonSources.Dialects.std
  TorchMLIRPythonSources
  TorchMLIRPythonExtensions
)

add_mlir_python_common_capi_library(TorchMLIRAggregateCAPI
  INSTALL_COMPONENT TorchMLIRPythonModules
  INSTALL_DESTINATION python_packages/torch_mlir/torch_mlir/_mlir_libs
  OUTPUT_DIRECTORY "${TORCH_MLIR_PYTHON_PACKAGES_DIR}/torch_mlir/torch_mlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES ${_source_components}
)

add_mlir_python_modules(TorchMLIRPythonModules
  ROOT_PREFIX "${TORCH_MLIR_PYTHON_PACKAGES_DIR}/torch_mlir/torch_mlir"
  INSTALL_PREFIX "python_packages/torch_mlir/torch_mlir"
  DECLARED_SOURCES ${_source_components}
  COMMON_CAPI_LINK_LIBS
    TorchMLIRAggregateCAPI
  )

# TODO: Find a cleaner way to do this.
# Can we build the JIT IR importer with `declare_mlir_python_extension`?
# Then it would "just work".
if(TORCH_MLIR_ENABLE_JIT_IR_IMPORTER)
  add_dependencies(TorchMLIRPythonModules TorchMLIRJITIRImporter)
endif()
