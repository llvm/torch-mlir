add_mlir_dialect_library(NPCOMPNpcomprt
  NpcomprtDialect.cpp
  NpcomprtOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/npcomp/Dialect/Npcomprt

  DEPENDS
  MLIRNpcomprtOpsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)
