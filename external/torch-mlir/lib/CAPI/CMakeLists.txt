add_mlir_library(TorchMLIRCAPI
  Registration.cpp
  TorchTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/torch-mlir-c/

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  TorchMLIRTorchDialect
  TorchMLIRInitAll
)

torch_mlir_target_includes(TorchMLIRCAPI)
