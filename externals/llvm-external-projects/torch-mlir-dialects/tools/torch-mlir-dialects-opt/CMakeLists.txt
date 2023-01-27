set(LIBS
  MLIRArithDialect
  MLIRDialect
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIROptLib
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRFuncDialect
  MLIRTensorDialect
  MLIRTransforms
  TorchMLIRDialectsConversionPasses
  TorchMLIRTMTensorDialect
  TorchMLIRTMTensorPasses
)

if(TORCH_MLIR_DIALECTS_ENABLE_TCP)
  list(APPEND LIBS TorchMLIRTcpDialect)
  list(APPEND LIBS TorchMLIRTcpPasses)
endif()

if(TORCH_MLIR_ENABLE_STABLEHLO)
  list(APPEND LIBS StablehloOps)
endif()

add_llvm_tool(torch-mlir-dialects-opt
  torch-mlir-dialects-opt.cpp

  DEPENDS
  ${LIBS}
)
target_link_libraries(torch-mlir-dialects-opt PRIVATE ${LIBS})
