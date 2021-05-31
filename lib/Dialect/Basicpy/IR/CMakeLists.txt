add_npcomp_dialect_library(NPCOMPBasicpyDialect
  BasicpyDialect.cpp
  BasicpyOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/npcomp/Dialect/Basicpy

  DEPENDS
  MLIRBasicpyOpsIncGen

  LINK_LIBS PUBLIC
  NPCOMPTypingCPA
  MLIRIR
  MLIRMemRef
  MLIRStandard
)
