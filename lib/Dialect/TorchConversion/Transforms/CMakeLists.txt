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
  list(APPEND MhloContract VerifyMhloBackendContract.cpp)
endif()

add_mlir_library(TorchMLIRTorchConversionPasses
  BackendTypeConversion.cpp
  BackendTypeConversionPasses.cpp  
  Passes.cpp
  VerifyLinalgOnTensorsBackendContract.cpp
  VerifyTosaBackendContract.cpp
  ${MhloContract}


  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Dialect/TorchConversion/Transforms

  DEPENDS
  TorchMLIRTorchConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  ${LinkedLibs}
)
