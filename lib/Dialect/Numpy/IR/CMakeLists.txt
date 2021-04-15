add_npcomp_dialect_library(NPCOMPNumpyDialect
  NumpyDialect.cpp
  NumpyOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/npcomp/Dialect/Numpy

  DEPENDS
  MLIRNumpyOpsIncGen

  LINK_LIBS PUBLIC
  NPCOMPBasicpyDialect
  NPCOMPTypingCPA
  NPCOMPTypingCPASupport
  MLIRIR
)
