add_npcomp_conversion_library(NPCOMPTCFToLinalg
  TCFToLinalg.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/npcomp/Conversion/TCFToLinalg

  DEPENDS
  NPCOMPConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRShape
  MLIRMemRef
  NPCOMPTCFDialect
  NPCOMPTCPDialect
)
