add_mlir_library(TorchMLIRTMTensorDialect
  TMTensorDialect.cpp
  TMTensorInterfaces.cpp
  TMTensorOps.cpp
  ScalarLoopOpInterface.cpp

  ADDITIONAL_HEADER_DIRS
  ${TORCH_MLIR_DIALECTS_SOURCE_DIR}/include

  DEPENDS
  TorchMLIRTMTensorOpsIncGen

  LINK_LIBS PUBLIC
  MLIRAffine
  MLIRDialectUtils
  MLIRIR
  MLIRLinalg
  MLIRMath
  MLIRMemRef
  MLIRPass
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRSCF
  MLIRFunc
  MLIRTensor
  MLIRViewLikeInterface
)

torch_mlir_dialects_target_includes(TorchMLIRTMTensorDialect)
