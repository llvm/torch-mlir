add_subdirectory(TorchOnnxToTorch)
add_subdirectory(TorchToArith)
add_subdirectory(TorchToLinalg)
add_subdirectory(TorchToSCF)
add_subdirectory(TorchToTensor)
if(TORCH_MLIR_ENABLE_TOSA)
  add_subdirectory(TorchToTosa)
endif()
if(TORCH_MLIR_ENABLE_STABLEHLO)
  add_subdirectory(TorchToStablehlo)
endif()
add_subdirectory(TorchToTMTensor)
add_subdirectory(TorchConversionToMLProgram)
add_subdirectory(Utils)

# TODO: Automate this with add_torch_mlir_conversion_library.
set(linked_libs TorchMLIRTorchToArith
                TorchMLIRTorchToLinalg
                TorchMLIRTorchToSCF
                TorchMLIRTorchToTensor
                TorchMLIRTorchToTMTensor
                TorchMLIRTorchConversionToMLProgram
                TorchMLIRConversionUtils)
if(TORCH_MLIR_ENABLE_STABLEHLO)
  list(APPEND linked_libs TorchMLIRTorchToStablehlo)
endif()
if(TORCH_MLIR_ENABLE_TOSA)
  list(APPEND linked_libs TorchMLIRTorchToTosa)
endif()

add_mlir_library(TorchMLIRConversionPasses
  Passes.cpp

  DEPENDS
  TorchMLIRConversionPassIncGen
  MLIRTorchOpsIncGen
  MLIRTorchTypesIncGen

  LINK_LIBS PUBLIC
  ${linked_libs}
)
