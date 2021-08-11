add_npcomp_dialect_library(NPCOMPTorchConversionDialect
  TorchConversionDialect.cpp
  TorchConversionOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/npcomp/Dialect/TorchConversion

  DEPENDS
  MLIRTorchConversionOpsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRSideEffectInterfaces
)
