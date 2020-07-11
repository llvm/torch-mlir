add_mlir_conversion_library(NPCOMPTCFPasses
  ShapeRefinement.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/npcomp/Dialect/TCF/Transforms

  DEPENDS
  NPCOMPTCFPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  NPCOMPTCF
)
