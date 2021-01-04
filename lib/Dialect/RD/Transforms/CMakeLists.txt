add_npcomp_conversion_library(NPCOMPRDPasses
  ExtractPipelineDef.cpp
  Passes.cpp
  MergeFuncs.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/npcomp/Dialect/RD/Transforms

  DEPENDS
  NPCOMPRDPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  NPCOMPRDDialect
)
