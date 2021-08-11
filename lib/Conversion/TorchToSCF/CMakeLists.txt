add_npcomp_conversion_library(NPCOMPTorchToSCF
  TorchToSCF.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/npcomp/Conversion/TorchToSCF

  DEPENDS
  NPCOMPConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSCF
  MLIRStandard
  NPCOMPTorchDialect
  NPCOMPTorchConversionDialect
)
