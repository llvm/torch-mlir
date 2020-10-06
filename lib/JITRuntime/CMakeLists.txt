add_mlir_library(NPCOMPJITRuntime
  JITModule.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SRC_DIR}/include/npcomp/JITRuntime

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  NPCOMPRuntime
  NPCOMPRefBackend
  MLIRExecutionEngine
  )

mlir_check_all_link_libraries(NPCOMPRefBackend)
