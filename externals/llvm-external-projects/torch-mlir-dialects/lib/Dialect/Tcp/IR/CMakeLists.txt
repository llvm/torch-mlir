add_mlir_library(TorchMLIRTcpDialect
  TcpDialect.cpp
  TcpOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${TORCH_MLIR_DIALECTS_SOURCE_DIR}/include

  DEPENDS
  TorchMLIRTcpOpsIncGen

  LINK_LIBS PUBLIC
  MLIRDialectUtils
  MLIRIR
  MLIRPass
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRViewLikeInterface
)

torch_mlir_dialects_target_includes(TorchMLIRTcpDialect)
