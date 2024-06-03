add_mlir_dialect_library(TorchMLIRTorchUtils
  Utils.cpp
  SparsityUtils.cpp
  TorchUpstream.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Dialect/Torch/Utils

  DEPENDS
  MLIRTorchOpsIncGen
  MLIRTorchTypesIncGen
)
