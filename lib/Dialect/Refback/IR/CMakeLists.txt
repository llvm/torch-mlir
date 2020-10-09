add_npcomp_dialect_library(NPCOMPRefbackDialect
  RefbackDialect.cpp
  RefbackOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/npcomp/Dialect/Refback

  DEPENDS
  MLIRRefbackOpsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRSideEffectInterfaces
  MLIRShape
  )
