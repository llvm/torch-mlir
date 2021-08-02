include(AddMLIRPython)
include(MLIRDetectPythonEnv)

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
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  SOURCES
    npcomp/__init__.py
    npcomp/build.py
    npcomp/decorators.py
    npcomp/exporter.py
    npcomp/smoketest.py
    npcomp/types.py
    npcomp/dialects/_ods_common.py
  SOURCES_GLOB
    npcomp/compiler/*.py
    npcomp/frontends/*.py
    npcomp/torch/*.py
    npcomp/tracing/*.py
    npcomp/utils/*.py
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
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  TD_FILE npcomp/dialects/BasicpyBind.td
  SOURCES npcomp/dialects/basicpy.py
  DIALECT_NAME basicpy)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT NPCOMPPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  TD_FILE npcomp/dialects/NumpyBind.td
  SOURCES npcomp/dialects/numpy.py
  DIALECT_NAME numpy)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT NPCOMPPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  TD_FILE npcomp/dialects/TorchBind.td
  SOURCES npcomp/dialects/torch.py
  DIALECT_NAME torch)

################################################################################
# Build composite binaries
################################################################################

# Bundle our own, self-contained CAPI library with all of our deps.
add_mlir_python_common_capi_library(NPCOMPPythonCAPI
  INSTALL_COMPONENT NPCOMPPythonModules
  INSTALL_DESTINATION python_packages/npcomp_core/mlir/_mlir_libs
  # NOTE: When the MLIR API is relocated under npcomp, this would change to
  # .../npcomp/_mlir_libs
  OUTPUT_DIRECTORY "${MLIR_NPCOMP_PYTHON_PACKAGES_DIR}/npcomp_core/mlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    # TODO: This can be chopped down significantly for size.
    MLIRPythonSources
    MLIRPythonExtension.AllPassesRegistration
    NPCOMPPythonSources
    NPCOMPPythonExtensions
)

# Bundle the MLIR python sources into our package.
# The MLIR API is position independent, so we explicitly output it to the mlir/
# folder as a temporary measure. It will eventually migrate under the npcomp/
# folder and be accessible under the unified "import npcomp..." namespace.
add_mlir_python_modules(NPCOMPMLIRPythonModules
  ROOT_PREFIX "${MLIR_NPCOMP_PYTHON_PACKAGES_DIR}/npcomp_core/mlir"
  INSTALL_PREFIX "python_packages/npcomp_core/mlir"
  DECLARED_SOURCES
    MLIRPythonSources
    MLIRPythonExtension.AllPassesRegistration
    # We need the npcomp extensions co-located with the MLIR extensions. When
    # the namespace is unified, this moves to the below.
    MLIRPythonCAPIHeaderSources
    NPCOMPPythonExtensions
    NPCOMPPythonCAPIHeaderSources
  COMMON_CAPI_LINK_LIBS
    NPCOMPPythonCAPI
)

# Bundle the NPCOMP python sources into our package.
add_mlir_python_modules(NPCOMPPythonModules
  ROOT_PREFIX "${MLIR_NPCOMP_PYTHON_PACKAGES_DIR}/npcomp_core"
  INSTALL_PREFIX "python_packages/npcomp_core"
  DECLARED_SOURCES
    NPCOMPPythonSources
  COMMON_CAPI_LINK_LIBS
    NPCOMPPythonCAPI
)
