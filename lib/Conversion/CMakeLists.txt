add_subdirectory(TorchOnnxToTorch)
add_subdirectory(TorchToArith)
add_subdirectory(TorchToLinalg)
add_subdirectory(TorchToSCF)
add_subdirectory(TorchToTensor)
add_subdirectory(TorchToTosa)
add_subdirectory(TorchToTosaLinalg)
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
                TorchMLIRTorchToTosa
                TorchMLIRTorchToTosaLinalg
                TorchMLIRTorchToTMTensor
                TorchMLIRTorchConversionToMLProgram
                TorchMLIRConversionUtils)
if(TORCH_MLIR_ENABLE_STABLEHLO)
  list(APPEND linked_libs TorchMLIRTorchToStablehlo)
endif()

add_mlir_library(TorchMLIRConversionPasses
  Passes.cpp

  DEPENDS
  TorchMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  ${linked_libs}
)
