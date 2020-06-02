add_mlir_dialect_library(NPCOMPNpcompRt
  NpcompRtDialect.cpp
  NpcompRtOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/npcomp/Dialect/NpcompRt

  DEPENDS
  MLIRNpcompRtOpsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)
