add_mlir_public_c_api_library(TorchMLIRCAPI
  Dialects.cpp
  Registration.cpp
  TorchOps.cpp
  TorchTypes.cpp
  Transforms.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/torch-mlir-c/

  ENABLE_AGGREGATION
  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRAffineDialect
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRTosaDialect
  MLIRAffineToStandard
  MLIRArithToLLVM
  MLIRArithTransforms
  MLIRBufferizationTransforms
  MLIRControlFlowToLLVM
  MLIRFuncToLLVM
  MLIRFuncTransforms
  MLIRLinalgToLLVM
  MLIRLinalgTransforms
  MLIRMathToLLVM
  MLIRMemRefToLLVM
  MLIRReconcileUnrealizedCasts
  MLIRSCFToControlFlow
  MLIRSCFTransforms
  MLIRTensorTransforms
  MLIRTosaToArith
  MLIRTosaToLinalg
  MLIRSupport
  TorchMLIRTorchDialect
  TorchMLIRInitAll
  TorchMLIRTorchPasses
)

torch_mlir_target_includes(TorchMLIRCAPI)
