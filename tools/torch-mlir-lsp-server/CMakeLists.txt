# torch-mlir-lsp-server is always linked dynamically as we want to distribute the
# binaries with the python packages for hacking/debugging.
add_executable(torch-mlir-lsp-server torch-mlir-lsp-server.cpp)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

target_link_libraries(torch-mlir-lsp-server PRIVATE
  MLIRLspServerLib
  TorchMLIRInitAll

  # TODO: Remove these in favor of interface deps.
  ${dialect_libs}
  ${conversion_libs}
)

mlir_check_all_link_libraries(torch-mlir-lsp-server)
