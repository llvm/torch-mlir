add_subdirectory(CAPI)
add_subdirectory(Conversion)
add_subdirectory(Dialect)
add_subdirectory(RefBackend)

if(TORCH_MLIR_ENABLE_IREE_DIALECTS)
  set(iree_dialect_libs IREELinalgExtDialect)
endif()

add_mlir_library(TorchMLIRInitAll
  InitAll.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport

  TorchMLIRTorchPasses
  TorchMLIRTorchConversionDialect

  TorchMLIRTorchDialect
  TorchMLIRTorchConversionPasses

  TorchMLIRConversionPasses
  TorchMLIRRefBackend

  ${iree_dialect_libs}
)

torch_mlir_target_includes(TorchMLIRInitAll)
