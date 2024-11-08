add_mlir_conversion_library(TorchMLIRTorchToLinalg
  DataMovement.cpp
  IndirectDataMovement.cpp
  Linear.cpp
  Pooling.cpp
  Random.cpp
  Reduction.cpp
  TensorConstructors.cpp
  TensorScalarInterop.cpp
  TorchToLinalg.cpp
  TorchvisionOp.cpp
  Uncategorized.cpp
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Conversion/TorchToLinalg

  DEPENDS
  TorchMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRLinalgDialect
  MLIRMathDialect
  TorchMLIRTorchDialect
)

torch_mlir_target_includes(TorchMLIRTorchToLinalg)
