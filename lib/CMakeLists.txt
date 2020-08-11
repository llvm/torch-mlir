add_subdirectory(Conversion)
add_subdirectory(Dialect)
add_subdirectory(E2E)
add_subdirectory(JITRuntime)
add_subdirectory(Python)
add_subdirectory(Typing)
add_subdirectory(runtime)

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

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_mlir_library(NPCOMPInitAll
  InitAll.cpp

  LINK_LIBS

  PUBLIC
  # Local depends
  NPCOMPE2E
  NPCOMPTCP
  NPCOMPTCF
  NPCOMPNpcomprt
  NPCOMPATenDialect
  NPCOMPBasicpyDialect
  NPCOMPBasicpyPasses
  NPCOMPNumpyDialect
  NPCOMPNumpyPasses
  NPCOMPTCFPasses
  NPCOMPTypingPasses

  ${dialect_libs}
  ${conversion_libs}
  ${ALL_DEPENDS}
)
