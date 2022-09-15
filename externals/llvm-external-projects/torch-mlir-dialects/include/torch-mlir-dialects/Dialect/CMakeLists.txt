add_subdirectory(TMTensor)
if(TORCH_MLIR_DIALECTS_ENABLE_TCP)
  add_subdirectory(Tcp)
endif()

