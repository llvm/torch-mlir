add_mlir_conversion_library(NPCOMPTypingPasses
  CPATypeInference.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/npcomp/Typing/Transforms

  DEPENDS
  NPCOMPTypingTransformsPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  NPCOMPTypingCPA
)
