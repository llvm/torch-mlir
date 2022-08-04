set(LinkedLibs  MLIRIR
                MLIRPass
                MLIRFuncTransforms
                TorchMLIRTorchConversionDialect
                TorchMLIRTorchDialect
                TorchMLIRTorchPasses
                TorchMLIRTorchToLinalg
                TorchMLIRTorchToTMTensor
                TorchMLIRTorchToArith
                TorchMLIRTorchToSCF
                MLIRMemRefTransforms)
                
if(TORCH_MLIR_ENABLE_MHLO)
  list(APPEND LinkedLibs ChloPasses)
endif()

add_mlir_library(TorchMLIRTorchConversionPasses
  BackendTypeConversion.cpp
  BackendTypeConversionPasses.cpp  
  Passes.cpp
  VerifyInvariantsBeforeBackendLowering.cpp
  VerifyLinalgOnTensorsBackendContract.cpp
  VerifyTosaBackendContract.cpp


  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Dialect/TorchConversion/Transforms

  DEPENDS
  TorchMLIRTorchConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  ${LinkedLibs}
)