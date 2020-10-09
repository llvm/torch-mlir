add_subdirectory(CAPI)
add_subdirectory(Conversion)
add_subdirectory(Dialect)
add_subdirectory(RefBackend)
add_subdirectory(Python)
add_subdirectory(Typing)

if(NPCOMP_ENABLE_REFJIT)
  add_subdirectory(Backend/RefJIT)
endif()
if(NPCOMP_ENABLE_IREE)
  add_subdirectory(Backend/IREE)
endif()

################################################################################
# Setup the initialization target.
# This includes conditional dependencies based on whether features are enabled.
################################################################################

set(ALL_DEPENDS)
if(NPCOMP_ENABLE_IREE)
  list(APPEND ALL_DEPENDS
    iree_tools_init_compiler_modules
    iree_tools_init_iree_passes_and_dialects
    iree_tools_init_mlir_passes_and_dialects
    iree_tools_init_targets
    iree_tools_init_xla_dialects
  )
endif()

get_property(mlir_dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(mlir_conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

get_property(npcomp_dialect_libs GLOBAL PROPERTY NPCOMP_DIALECT_LIBS)
get_property(npcomp_conversion_libs GLOBAL PROPERTY NPCOMP_CONVERSION_LIBS)

message(STATUS "NPCOMP Dialect libs: ${npcomp_dialect_libs}")
message(STATUS "NPCOMP Conversion libs: ${npcomp_conversion_libs}")

add_npcomp_library(NPCOMPInitAll
  InitAll.cpp

  LINK_LIBS

  PUBLIC
  # Local depends
  NPCOMPRefBackend
  NPCOMPRefbackDialect
  NPCOMPTCPDialect
  NPCOMPTCFDialect
  NPCOMPTorchDialect
  NPCOMPRefbackrtDialect
  NPCOMPATenDialect
  NPCOMPBasicpyDialect
  NPCOMPBasicpyPasses
  NPCOMPConversionPasses
  NPCOMPNumpyDialect
  NPCOMPNumpyPasses
  NPCOMPTCFPasses
  NPCOMPTypingPasses

  ${npcomp_dialect_libs}
  ${npcomp_conversion_libs}
  ${mlir_dialect_libs}
  ${mlir_conversion_libs}
  ${ALL_DEPENDS}
)
