add_subdirectory(CAPI)
add_subdirectory(Conversion)
add_subdirectory(Dialect)

set(LinkedLibs
  MLIRComplexDialect
  MLIRFuncDialect
  MLIRFuncInlinerExtension
  MLIRIR
  MLIRMLProgramDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRTosaDialect
  MLIRSupport

  TorchMLIRTorchPasses
  TorchMLIRTorchConversionDialect

  TorchMLIRTorchDialect
  TorchMLIRTorchConversionPasses

  TorchMLIRTMTensorPasses
  TorchMLIRTMTensorDialect

  TorchMLIRConversionPasses
)

if(TORCH_MLIR_ENABLE_REFBACKEND)
  add_subdirectory(RefBackend)
  list(APPEND LinkedLibs TorchMLIRRefBackend)
endif()

add_mlir_library(TorchMLIRInitAll
  InitAll.cpp

  LINK_LIBS PUBLIC
  ${LinkedLibs}
)

torch_mlir_target_includes(TorchMLIRInitAll)
