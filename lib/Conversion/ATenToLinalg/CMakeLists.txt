add_npcomp_conversion_library(NPCOMPATenToLinalg
  ATenToLinalg.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/npcomp/Conversion/ATenToLinalg

  DEPENDS
  NPCOMPConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  NPCOMPATenDialect
  MLIRLinalg
)
