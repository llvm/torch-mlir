add_mlir_library(TorchMLIRTorchPasses
  AdjustCallingConventions.cpp
  DecomposeComplexOps.cpp
  DecomposeComplexOpsEarly.cpp
  DropShapeCalculations.cpp
  Passes.cpp
  GlobalizeObjectGraph.cpp
  InlineGlobalSlots.cpp
  MaximizeValueSemantics.cpp
  PrepareForGlobalizeObjectGraph.cpp
  ReduceOpVariants.cpp
  RefinePublicReturn.cpp
  RefineTypes.cpp
  ReifyShapeCalculations.cpp
  ShapeLibrary.cpp
  SimplifyShapeCalculations.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Dialect/Torch/Transforms

  DEPENDS
  TorchMLIRTorchPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  TorchMLIRTorchDialect
  TorchMLIRTorchUtils
)

torch_mlir_target_includes(TorchMLIRTorchPasses)
