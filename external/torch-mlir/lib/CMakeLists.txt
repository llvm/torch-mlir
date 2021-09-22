add_subdirectory(CAPI)
add_subdirectory(Dialect)
add_subdirectory(RefBackend)

add_mlir_library(TorchMLIRInitAll
  InitAll.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  TorchMLIRTorchDialect
  TorchMLIRTorchPasses
  TorchMLIRRefBackend
)

torch_mlir_target_includes(TorchMLIRInitAll)
