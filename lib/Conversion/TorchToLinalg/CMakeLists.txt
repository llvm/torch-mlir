add_mlir_conversion_library(TorchMLIRTorchToLinalg
  Passes.cpp
  DataMovement.cpp
  IndirectDataMovement.cpp
  Linear.cpp
  Pooling.cpp
  Random.cpp
  Reduction.cpp
  TensorConstructors.cpp
  TensorScalarInterop.cpp
  TorchConversionToMLProgram.cpp
  TorchToLinalg.cpp
  TorchToTMTensor.cpp
  Uncategorized.cpp
  Utils.cpp
  VerifyLinalgOnTensorsBackendContract.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Conversion/TorchToLinalg

  DEPENDS
  TorchMLIRConversionLinalgPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRFuncDialect
  MLIRPass
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRMLProgramDialect
  MLIRSCFDialect
  MLIRTransforms
  TorchMLIRConversionCommon
  TorchMLIRTorchDialect
  TorchMLIRTorchConversionDialect
  TorchMLIRTorchConversionPasses
  TorchMLIRTorchUtils
  TorchMLIRTMTensorDialect
)

torch_mlir_target_includes(TorchMLIRTorchToLinalg)
