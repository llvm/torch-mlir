add_mlir_library(TorchMLIRTorchPasses
  AdjustCallingConventions.cpp
  DecomposeComplexOps.cpp
  DropAbstractInterpCalculations.cpp
  EraseModuleInitializer.cpp
  FuseQuantizedOps.cpp
  Passes.cpp
  GlobalizeObjectGraph.cpp
  InlineGlobalSlots.cpp
  LowerToBackendContract.cpp
  MatchQuantizedOps.cpp
  MaximizeValueSemantics.cpp
  PrepareForGlobalizeObjectGraph.cpp
  RecomposeComplexOps.cpp
  ReduceOpVariants.cpp
  RefinePublicReturn.cpp
  ReifyShapeCalculations.cpp
  ReifyDtypeCalculations.cpp
  ReifyAbstractInterpCalculationsUtils.cpp
  AbstractInterpLibrary.cpp
  SimplifyShapeCalculations.cpp
  SimplifyDtypeCalculations.cpp
  SimplifyAbstractInterpCalculationsUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Dialect/Torch/Transforms

  DEPENDS
  TorchMLIRTorchPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  TorchMLIRTorchDialect
  TorchMLIRTorchUtils
)

torch_mlir_target_includes(TorchMLIRTorchPasses)
