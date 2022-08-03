add_mlir_conversion_library(TorchMLIRTorchToMhlo
  TorchToMhlo.cpp
  MhloLegalizeUtils.cpp
  BasicOp.cpp
  GatherOp.cpp
  Linear.cpp
  ViewLikeOps.cpp
  ReductionOp.cpp
  PoolingOp.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Conversion/TorchToMhlo

  DEPENDS
  MhloDialect
  ChloDialect
  TorchMLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MhloDialect
  ChloDialect
  TorchMLIRTorchDialect
)

torch_mlir_target_includes(TorchMLIRTorchToMhlo)
