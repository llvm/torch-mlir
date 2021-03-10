add_npcomp_conversion_library(NPCOMPTorchPasses
  Passes.cpp
  GlobalizeObjectGraph.cpp
  PrepareForGlobalizeObjectGraph.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/npcomp/Dialect/Torch/Transforms

  DEPENDS
  NPCOMPTorchPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  NPCOMPTorchDialect
  NPCOMPBasicpyDialect
)
