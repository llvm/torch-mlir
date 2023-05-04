add_mlir_library(TorchMLIRTcpPasses
  FuseTcpOpsPass.cpp
  IsolateGroupOpsPass.cpp
  Passes.cpp

  DEPENDS
  TorchMLIRTcpPassIncGen

  LINK_LIBS PUBLIC
  TorchMLIRTcpDialect
  MLIRIR
  MLIRPass
  MLIRFuncDialect
  MLIRSupport
  MLIRTransforms
)
