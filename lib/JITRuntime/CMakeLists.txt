add_mlir_library(NPCOMPJITRuntime
  JITModule.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SRC_DIR}/include/npcomp/JITRuntime

  DEPENDS
  NPCOMPE2E
  MLIRExecutionEngine

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  NPCOMPRuntime
  )
