add_mlir_library(TorchMLIRRefBackend
  RefBackend.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SRC_DIR}/include/torch-mlir/RefBackend

  DEPENDS
  MLIRTorchTypesIncGen
  TorchMLIRRefBackendPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRTransforms
  MLIRMathTransforms
  )

mlir_check_all_link_libraries(TorchMLIRRefBackend)
torch_mlir_target_includes(TorchMLIRRefBackend)
