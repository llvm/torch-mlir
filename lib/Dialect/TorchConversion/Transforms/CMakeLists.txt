set(LinkedLibs
  MLIRFuncTransforms
  MLIRIR
  MLIRPass
  TorchMLIRTorchConversionDialect
  TorchMLIRTorchDialect
  TorchMLIRTorchPasses
  )

if(TORCH_MLIR_ENABLE_STABLEHLO)
  list(APPEND LinkedLibs ChloPasses)
endif()

add_mlir_library(TorchMLIRTorchConversionPasses
  BackendTypeConversion.cpp
  BackendTypeConversionPasses.cpp  
  Passes.cpp
  ConvertCustomQuantOp.cpp
  UnpackQuantTensor.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Dialect/TorchConversion/Transforms

  DEPENDS
  TorchMLIRTorchConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  ${LinkedLibs}
)
