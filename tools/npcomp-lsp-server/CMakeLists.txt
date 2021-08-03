# npcomp-lsp-server is always linked dynamically as we want to distribute the
# binaries with the python packages for hacking/debugging.
add_npcomp_executable(npcomp-lsp-server npcomp-lsp-server.cpp)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

target_link_libraries(npcomp-lsp-server PRIVATE
  MLIRLspServerLib
  NPCOMPInitAll

  # TODO: Remove these in favor of interface deps.
  ${dialect_libs}
  ${conversion_libs}
)

mlir_check_all_link_libraries(npcomp-lsp-server)
