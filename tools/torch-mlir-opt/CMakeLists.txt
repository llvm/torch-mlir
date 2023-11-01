add_llvm_executable(torch-mlir-opt torch-mlir-opt.cpp)

install(TARGETS torch-mlir-opt
EXPORT TorchMLIRTargets
RUNTIME DESTINATION ${LLVM_TOOLS_INSTALL_DIR}
COMPONENT torch-mlir-opt)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

set(dependency_libraries)
if(TORCH_MLIR_ENABLE_STABLEHLO)
  list(APPEND dependency_libraries StablehloRegister)
endif()

target_link_libraries(torch-mlir-opt PRIVATE
  MLIROptLib
  TorchMLIRInitAll
  TorchMLIRTorchDialect
  TorchMLIRTorchPasses
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
  ${dependency_libraries}
)
