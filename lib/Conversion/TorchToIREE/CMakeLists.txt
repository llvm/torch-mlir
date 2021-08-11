add_npcomp_conversion_library(NPCOMPTorchToIREE
  TorchToIREE.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/npcomp/Conversion/TorchToIREE

  DEPENDS
  NPCOMPConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  NPCOMPTorchDialect
  MLIRStandard
  IREEDialectsIREEDialect
)
