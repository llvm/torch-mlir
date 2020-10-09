add_npcomp_dialect_library(NPCOMPRefbackrtDialect
  RefbackrtDialect.cpp
  RefbackrtOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/npcomp/Dialect/Refbackrt

  DEPENDS
  MLIRRefbackrtOpsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)
