add_llvm_executable(torch-mlir-opt torch-mlir-opt.cpp)

install(TARGETS torch-mlir-opt
EXPORT TorchMLIRTargets
RUNTIME DESTINATION ${LLVM_TOOLS_INSTALL_DIR}
COMPONENT torch-mlir-opt)

set(dependency_libraries)
if(TORCH_MLIR_ENABLE_STABLEHLO)
  list(APPEND dependency_libraries StablehloRegister)
endif()

target_link_libraries(torch-mlir-opt PRIVATE
  MLIROptLib
  MLIRTransforms
  TorchMLIRInitAll
  TorchMLIRTorchDialect
  TorchMLIRTorchPasses
  ${dependency_libraries}
)
