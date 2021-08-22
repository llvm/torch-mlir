include(AddMLIRPython)
include(MLIRDetectPythonEnv)

# Specifies that all MLIR packages are co-located under npcomp.
# TODO: Add an upstream cmake param for this vs having a global here.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=npcomp.")

################################################################################
# Resources that must be packaged into the python tree
################################################################################

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/npcomp/compiler/backend/refjit_resources")
add_custom_target(NPCOMPPythonResources ALL)
add_custom_command(
  TARGET NPCOMPPythonResources
  COMMAND ${CMAKE_COMMAND} -E copy
        # TODO: Make the runtime library work for windows.
        # TODO: Use $<TARGET-FILE:> for this.
        ${CMAKE_BINARY_DIR}/lib/libNPCOMPCompilerRuntimeShlib${CMAKE_SHARED_LIBRARY_SUFFIX}
        ${MLIR_NPCOMP_PYTHON_PACKAGES_DIR}/npcomp_core/npcomp/compiler/generic/backend/libNPCOMPCompilerRuntimeShlib${CMAKE_SHARED_LIBRARY_SUFFIX}
)
add_dependencies(NPCOMPPythonResources
  NPCOMPCompilerRuntimeShlib
  )


################################################################################
# Declare sources
################################################################################

declare_mlir_python_sources(NPCOMPPythonSources)

declare_mlir_python_sources(NPCOMPPythonSources.Core
  ADD_TO_PARENT NPCOMPPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/npcomp"
  SOURCES
    build.py
    decorators.py
    exporter.py
    smoketest.py
    types.py
  SOURCES_GLOB
    compiler/*.py
    frontends/*.py
    torch/*.py
    utils/*.py
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

set(_addl_extension_sources)
if(NPCOMP_ENABLE_REFJIT)
  list(APPEND _addl_extension_sources "${CMAKE_CURRENT_SOURCE_DIR}/RefJITBackend.cpp")
endif()

declare_mlir_python_extension(NPCOMPPythonExtensions.Core
  MODULE_NAME _npcomp
  ADD_TO_PARENT NPCOMPPythonExtensions
  SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/NpcompModule.cpp
    ${_addl_extension_sources}
  EMBED_CAPI_LINK_LIBS
    NPCOMPCAPI
  PRIVATE_LINK_LIBS
    LLVMSupport
)

################################################################################
# Declare dialects
################################################################################

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT NPCOMPPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/npcomp"
  TD_FILE dialects/BasicpyBind.td
  SOURCES dialects/basicpy.py
  DIALECT_NAME basicpy)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT NPCOMPPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/npcomp"
  TD_FILE dialects/NumpyBind.td
  SOURCES dialects/numpy.py
  DIALECT_NAME numpy)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT NPCOMPPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/npcomp"
  TD_FILE dialects/TorchBind.td
  SOURCES dialects/torch.py
  DIALECT_NAME torch)

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
