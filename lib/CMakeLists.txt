add_subdirectory(CAPI)
add_subdirectory(Conversion)
add_subdirectory(Dialect)
add_subdirectory(RefBackend)

get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
set(LinkedLibs
  MLIRFuncDialect
  MLIRIR
  MLIRSupport
  ${extension_libs}

  TorchMLIRTorchPasses
  TorchMLIRTorchConversionDialect

  TorchMLIRTorchDialect
  TorchMLIRTorchConversionPasses

  TorchMLIRTMTensorPasses
  TorchMLIRTMTensorDialect

  TorchMLIRConversionPasses
  TorchMLIRRefBackend
)

add_mlir_library(TorchMLIRInitAll
  InitAll.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  ${LinkedLibs}
)

torch_mlir_target_includes(TorchMLIRInitAll)
