add_npcomp_conversion_library(NPCOMPTorchToStd
  TorchToStd.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/npcomp/Conversion/TorchToStd

  DEPENDS
  NPCOMPConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRStandard
  MLIRTransformUtils
  NPCOMPTorchDialect
)
