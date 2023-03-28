set(linked_libs TorchMLIRConversionUtils)

if(TORCH_MLIR_DIALECTS_ENABLE_TCP)
  if(TORCH_MLIR_ENABLE_STABLEHLO)
    add_subdirectory(StablehloToTcp)
    list(APPEND linked_libs TorchMLIRStablehloToTcp)
  endif()
  add_subdirectory(TcpToArith)
  add_subdirectory(TcpToLinalg)
  list(APPEND linked_libs TorchMLIRTcpToArith)
  list(APPEND linked_libs TorchMLIRTcpToLinalg)
endif()

add_mlir_library(TorchMLIRDialectsConversionPasses
  Passes.cpp

  DEPENDS
  TorchMLIRDialectsConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  ${linked_libs}
)
