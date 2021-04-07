add_npcomp_conversion_library(NPCOMPTCFToTCP
  TCFToTCP.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/npcomp/Conversion/TCFToTCP

  DEPENDS
  NPCOMPConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRShape
  NPCOMPTCFDialect
  NPCOMPTCPDialect
)
