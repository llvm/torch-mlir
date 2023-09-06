add_subdirectory(CAPI)
add_subdirectory(Conversion)
add_subdirectory(Dialect)
add_subdirectory(RefBackend)

get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
set(LinkedLibs
  MLIRFuncDialect
  MLIRIR
  MLIRSupport
  ${extension_libs}

  TorchMLIRTorchPasses
  TorchMLIRTorchConversionDialect

  TorchMLIRTorchDialect
  TorchMLIRTorchConversionPasses

  TorchMLIRTMTensorPasses
  TorchMLIRTMTensorDialect

  TorchMLIRConversionPasses
  TorchMLIRRefBackend
)

# Conditionally link in backends if enabled.
if(TORCH_MLIR_ENABLE_LINALG)
  list(APPEND LinkedLibs TorchMLIRTorchToLinalg)
endif()
if(TORCH_MLIR_ENABLE_TOSA)
  list(APPEND LinkedLibs TorchMLIRTorchToTosa)
endif()
if(TORCH_MLIR_ENABLE_STABLEHLO)
  list(APPEND LinkedLibs
    TorchMLIRTorchToStablehlo
    MhloPasses
    MhloToLinalg
    StablehloToMhlo
  )
endif()

add_mlir_library(TorchMLIRInitAll
  InitAll.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  ${LinkedLibs}
)

torch_mlir_target_includes(TorchMLIRInitAll)
