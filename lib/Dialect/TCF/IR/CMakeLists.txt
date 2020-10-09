add_npcomp_dialect_library(NPCOMPTCFDialect
  TCFDialect.cpp
  TCFOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/npcomp/Dialect/TCF

  DEPENDS
  MLIRTCFOpsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)
