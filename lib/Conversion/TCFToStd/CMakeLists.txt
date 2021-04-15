add_npcomp_conversion_library(NPCOMPTCFToStd
  TCFToStd.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/npcomp/Conversion/TCFToStd

  DEPENDS
  NPCOMPConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRShape
  MLIRStandard
  MLIRLinalg
  MLIRMath
  NPCOMPTCFDialect
  NPCOMPTCPDialect
)
