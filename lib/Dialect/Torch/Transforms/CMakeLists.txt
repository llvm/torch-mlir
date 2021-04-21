add_npcomp_conversion_library(NPCOMPTorchPasses
  AdjustCallingConventions.cpp
  Passes.cpp
  GlobalizeObjectGraph.cpp
  PrepareForGlobalizeObjectGraph.cpp
  RefineTypes.cpp

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
  NPCOMPATenPasses
  NPCOMPNumpyPasses
  NPCOMPATenToLinalg
  NPCOMPATenToTCF
  NPCOMPTCFToStd
)
