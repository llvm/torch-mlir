add_npcomp_library(NPCOMPRefBackendJITHelpers
  JITModule.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SRC_DIR}/include/npcomp/RefBackend/JITHelpers

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  NPCOMPRuntime
  NPCOMPRefBackend
  MLIRExecutionEngine
  )

mlir_check_all_link_libraries(NPCOMPRefBackend)
