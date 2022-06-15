add_mlir_conversion_library(TorchMLIRTorchToLinalg
# TODO: Re-enable after MacOS support is fixed for the custom op extension.
#  CustomOpExample.cpp
  DataMovement.cpp
  IndirectDataMovement.cpp
  Linear.cpp
  Pooling.cpp
  Random.cpp
  Reduction.cpp
  TensorConstructors.cpp
  TensorScalarInterop.cpp
  TorchToLinalg.cpp
  Uncategorized.cpp
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Conversion/TorchToLinalg

  DEPENDS
  TorchMLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRLinalg
  MLIRMath
  TorchMLIRTorchDialect
)

torch_mlir_target_includes(TorchMLIRTorchToLinalg)
