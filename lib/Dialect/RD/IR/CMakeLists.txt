add_npcomp_dialect_library(NPCOMPRDDialect
  RDDialect.cpp
  RDOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/npcomp/Dialect/RD

  DEPENDS
  MLIRRDOpsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRSideEffectInterfaces
  MLIRShape
  )
