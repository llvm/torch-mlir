set(LinkedLibs
  MLIRFuncTransforms
  MLIRIR
  MLIRLinalgTransforms
  MLIRMemRefTransforms
  MLIRPass
  MLIRTosaTransforms
  MLIRVectorTransforms
  TorchMLIRTorchConversionDialect
  TorchMLIRTorchConversionToMLProgram
  TorchMLIRTorchDialect
  TorchMLIRTorchPasses
  TorchMLIRTorchToArith
  TorchMLIRTorchToLinalg
  TorchMLIRTorchToSCF
  TorchMLIRTorchToTMTensor
  TorchMLIRTorchToTosa
  )

if(TORCH_MLIR_ENABLE_STABLEHLO)
  list(APPEND LinkedLibs ChloPasses)
endif()

add_mlir_library(TorchMLIRTorchConversionPasses
  BackendTypeConversion.cpp
  BackendTypeConversionPasses.cpp  
  Passes.cpp
  UnpackTensor.cpp
  VerifyLinalgOnTensorsBackendContract.cpp
  VerifyTosaBackendContract.cpp
  VerifyStablehloBackendContract.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Dialect/TorchConversion/Transforms

  DEPENDS
  TorchMLIRTorchConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  ${LinkedLibs}
)
