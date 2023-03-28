add_subdirectory(TorchToLinalg)
add_subdirectory(TorchToSCF)
add_subdirectory(TorchToArith)
add_subdirectory(TorchToTosa)
if(TORCH_MLIR_ENABLE_STABLEHLO)
  add_subdirectory(TorchToStablehlo)
endif()
if (TORCH_MLIR_ENABLE_TCP)
  add_subdirectory(TorchToTcp)
endif()
add_subdirectory(TorchToTMTensor)
add_subdirectory(TorchConversionToMLProgram)
add_subdirectory(Utils)

# TODO: Automate this with add_torch_mlir_conversion_library.
set(linked_libs TorchMLIRTorchToLinalg
                TorchMLIRTorchToSCF
                TorchMLIRTorchToArith
                TorchMLIRTorchToTosa
                TorchMLIRTorchToTMTensor
                TorchMLIRTorchConversionToMLProgram
                TorchMLIRConversionUtils)
if(TORCH_MLIR_ENABLE_STABLEHLO)
  list(APPEND linked_libs TorchMLIRTorchToStablehlo)
endif()
if(TORCH_MLIR_ENABLE_TCP)
  if(TORCH_MLIR_ENABLE_STABLEHLO)
    list(APPEND linked_libs TorchMLIRStablehloToTcp)
  endif()
  list(APPEND linked_libs TorchMLIRTorchToTcp)
  list(APPEND linked_libs TorchMLIRTcpToLinalg)
  list(APPEND linked_libs TorchMLIRTcpToArith)
endif()

add_mlir_library(TorchMLIRConversionPasses
  Passes.cpp

  DEPENDS
  TorchMLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  ${linked_libs}
  #${torch_mlir_conversion_libs}
)
