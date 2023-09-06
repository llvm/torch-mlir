if(TORCH_MLIR_ENABLE_LINALG)
  add_subdirectory(TorchToLinalg)
endif()
if(TORCH_MLIR_ENABLE_TOSA)
  add_subdirectory(TorchToTosa)
endif()
if(TORCH_MLIR_ENABLE_STABLEHLO)
  add_subdirectory(TorchToStablehlo)
endif()
add_subdirectory(Utils)

add_mlir_library(TorchMLIRConversionPasses
  Passes.cpp
  LINK_LIBS PUBLIC
    TorchMLIRConversionUtils
)

add_subdirectory(Common)