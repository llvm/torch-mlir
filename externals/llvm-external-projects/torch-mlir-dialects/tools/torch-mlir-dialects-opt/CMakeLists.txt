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
  TorchMLIRTMTensorDialect
  TorchMLIRTMTensorPasses
)

add_llvm_tool(torch-mlir-dialects-opt
  torch-mlir-dialects-opt.cpp

  DEPENDS
  ${LIBS}
)
target_link_libraries(torch-mlir-dialects-opt PRIVATE ${LIBS})
