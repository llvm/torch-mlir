add_executable(torch-mlir-opt torch-mlir-opt.cpp)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

target_link_libraries(torch-mlir-opt PRIVATE
  MLIROptLib
  TorchMLIRInitAll
  TorchMLIRTorchDialect
  TorchMLIRTorchPasses
  ${dialect_libs}
  ${conversion_libs}
)
