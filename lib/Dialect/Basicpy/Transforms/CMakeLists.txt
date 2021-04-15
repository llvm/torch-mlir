add_npcomp_conversion_library(NPCOMPBasicpyPasses
  Passes.cpp
  TypeInference.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/npcomp/Dialect/Basicpy/Transforms

  DEPENDS
  NPCOMPBasicpyPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  NPCOMPBasicpyDialect
  NPCOMPTypingCPA
)
