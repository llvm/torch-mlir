torch_mlir_enable_werror()

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
  MLIRTensorInferTypeOpInterfaceImpl
  MLIRSupport

  # Dialects.
  TorchMLIRTMTensorDialect
  TorchMLIRTorchDialect
  TorchMLIRTorchConversionDialect

  # Dialect passes.
  TorchMLIRTMTensorPasses
  TorchMLIRTorchConversionPasses
  TorchMLIRTorchPasses

  # Conversion passes.
  TorchMLIRConversionPasses
  TorchMLIRTorchOnnxToTorch
)

if(TORCH_MLIR_ENABLE_STABLEHLO)
  list(APPEND LinkedLibs StablehloLinalgTransforms StablehloPasses)
endif()

if (TORCH_MLIR_ENABLE_TOSA)
  list(APPEND LinkedLibs MLIRTosaDialect)
endif()

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
