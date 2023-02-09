add_mlir_conversion_library(TorchMLIRTcpToArith
  TcpToArith.cpp

  ADDITIONAL_HEADER_DIRS
  ${TORCH_MLIR_DIALECTS_SOURCE_DIR}/include

  DEPENDS
  TorchMLIRDialectsConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRDialectUtils
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRArithDialect
  MLIRSupport
  TorchMLIRTcpDialect
)

torch_mlir_dialects_target_includes(TorchMLIRTcpToArith)
