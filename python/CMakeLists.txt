include(AddMLIRPython)
include(MLIRDetectPythonEnv)

add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=npcomp.")

################################################################################
# Declare sources
################################################################################

declare_mlir_python_sources(NPCOMPPythonSources)

declare_mlir_python_sources(NPCOMPPythonSources.Core
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/npcomp"
  ADD_TO_PARENT NPCOMPPythonSources
  SOURCES_GLOB
    *.py
)

declare_mlir_python_sources(NPCOMPPythonSources.Dialects
  ADD_TO_PARENT NPCOMPPythonSources
)
declare_mlir_python_sources(NPCOMPPythonExtensions)

declare_mlir_python_sources(NPCOMPPythonCAPIHeaderSources
  ROOT_DIR "${MLIR_NPCOMP_SOURCE_DIR}/include"
  SOURCES_GLOB "npcomp-c/*.h"
  DEST_PREFIX "_mlir_libs/include"
)

################################################################################
# Extensions
################################################################################

declare_mlir_python_extension(NPCOMPPythonExtensions.Core
  MODULE_NAME _npcomp
  ADD_TO_PARENT NPCOMPPythonExtensions
  SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/NpcompModule.cpp
  EMBED_CAPI_LINK_LIBS
    NPCOMPCAPI
  PRIVATE_LINK_LIBS
    LLVMSupport
)

################################################################################
# Declare dialects
################################################################################

# Declare dialects with `declare_mlir_dialect_python_bindings`.
# Currently there are none, because the interesting ones to bind have been
# removed or moved into torch-mlir.

################################################################################
# Build composite binaries
################################################################################

# Bundle our own, self-contained CAPI library with all of our deps.
add_mlir_python_common_capi_library(NPCOMPPythonCAPI
  INSTALL_COMPONENT NPCOMPPythonModules
  INSTALL_DESTINATION python_packages/npcomp_core/npcomp/_mlir_libs
  OUTPUT_DIRECTORY "${MLIR_NPCOMP_PYTHON_PACKAGES_DIR}/npcomp_core/npcomp/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    # TODO: Common MLIR deps can be reduced substantially.
    MLIRPythonSources.Core
    MLIRPythonSources.Dialects
    MLIRPythonSources.ExecutionEngine
    MLIRPythonExtension.AllPassesRegistration
    NPCOMPPythonSources
    NPCOMPPythonExtensions
)

# Bundle MLIR and NPCOMP into the top-level npcomp package.
add_mlir_python_modules(NPCOMPPythonModules
  ROOT_PREFIX "${MLIR_NPCOMP_PYTHON_PACKAGES_DIR}/npcomp_core/npcomp"
  INSTALL_PREFIX "python_packages/npcomp_core/npcomp"
  DECLARED_SOURCES
    MLIRPythonSources
    MLIRPythonExtension.AllPassesRegistration
    MLIRPythonCAPIHeaderSources
    NPCOMPPythonSources
    NPCOMPPythonExtensions
    NPCOMPPythonCAPIHeaderSources
  COMMON_CAPI_LINK_LIBS
    NPCOMPPythonCAPI
)

################################################################################
# Torch support libraries.
################################################################################

# Torch-related modules that are too broad of a scope/opinion for torch-mlir
# belong there.
declare_mlir_python_sources(NPCOMPTorchSupportPythonSources
  ADD_TO_PARENT NPCOMPPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/torch_support"
  SOURCES_GLOB
    npcomp_torchscript/*.py
    torch_mlir_e2e_test.torchscript.configs/*.py
)

add_mlir_python_modules(NPCOMPTorchSupportPythonModules
  ROOT_PREFIX "${MLIR_NPCOMP_PYTHON_PACKAGES_DIR}/npcomp_core"
  INSTALL_PREFIX "python_packages/npcomp_core"
  DECLARED_SOURCES
    NPCOMPTorchSupportPythonSources
)
add_dependencies(NPCOMPTorchSupportPythonModules
  NPCOMPPythonModules
  )



################################################################################
# Recurse into the tests.
################################################################################
