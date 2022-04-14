add_subdirectory(TorchToLinalg)
add_subdirectory(TorchToSCF)
add_subdirectory(TorchToStd)
add_subdirectory(TorchToTosa)
add_subdirectory(TorchToMhlo)
add_subdirectory(TorchToTMTensor)
add_subdirectory(Utils)

# TODO: Automate this with add_torch_mlir_conversion_library.
#get_property(torch_mlir_conversion_libs GLOBAL PROPERTY TORCH_MLIR_CONVERSION_LIBS)

add_mlir_library(TorchMLIRConversionPasses
  Passes.cpp

  DEPENDS
  TorchMLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  TorchMLIRTorchToLinalg
  TorchMLIRTorchToSCF
  TorchMLIRTorchToStd
  TorchMLIRTorchToTosa
  TorchMLIRTorchToMhlo
  TorchMLIRTorchToTMTensor
  TorchMLIRConversionUtils
  #${torch_mlir_conversion_libs}
)
