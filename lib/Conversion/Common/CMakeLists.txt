add_mlir_conversion_library(TorchMLIRConversionCommon
  TorchToArith.cpp
  TorchToSCF.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Conversion

  DEPENDS
  TorchMLIRConversionCommonPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRIR
  MLIRFuncDialect
  MLIRPass
  MLIRMathDialect
  MLIRSCFDialect
  MLIRTransforms
  TorchMLIRTorchDialect
  TorchMLIRTorchConversionDialect
  TorchMLIRTorchConversionPasses
  TorchMLIRTorchUtils
)

torch_mlir_target_includes(TorchMLIRConversionCommon)
