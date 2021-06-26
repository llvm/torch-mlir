add_npcomp_conversion_library(NPCOMPTorchPasses
  AdjustCallingConventions.cpp
  BackendTypeConversion.cpp
  Passes.cpp
  GlobalizeObjectGraph.cpp
  InlineGlobalSlots.cpp
  MaximizeValueSemantics.cpp
  PrepareForGlobalizeObjectGraph.cpp
  ReduceOpVariants.cpp
  RefinePublicReturn.cpp
  RefineTypes.cpp
  VerifyInvariantsBeforeBackendLowering.cpp

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
  NPCOMPTorchToLinalg
  NPCOMPInterfaces
  MLIRMemRefTransforms
)
