set(LinkedLibs
  MLIRFuncTransforms
  MLIRControlFlowTransforms
  MLIRIR
  MLIRLinalgTransforms
  MLIRMemRefTransforms
  MLIRPass
  MLIRTosaTransforms
  MLIRVectorTransforms
  TorchMLIRTorchConversionDialect
  TorchMLIRTorchDialect
  TorchMLIRTorchPasses
  TorchMLIRConversionPasses
  )

if(TORCH_MLIR_ENABLE_STABLEHLO)
  list(APPEND LinkedLibs
    StablehloOps
    StablehloPasses
  )
endif()

add_mlir_library(TorchMLIRTorchConversionPasses
  BackendTypeConversion.cpp
  BackendTypeConversionPasses.cpp
  Passes.cpp
  ConvertCustomQuantOp.cpp
  UnpackQuantTensor.cpp
  VerifyLinalgOnTensorsBackendContract.cpp
  VerifyTosaBackendContract.cpp
  VerifyStablehloBackendContract.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Dialect/TorchConversion/Transforms

  DEPENDS
  TorchMLIRTorchConversionPassIncGen

  LINK_LIBS PUBLIC
  ${LinkedLibs}
)
