set(LLVM_LINK_COMPONENTS
  Core
  Support
  )

add_llvm_executable(npcomp-capi-ir-test
  ir.c
  )
llvm_update_compile_flags(npcomp-capi-ir-test)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
target_link_libraries(npcomp-capi-ir-test
  PRIVATE
  NPCOMPCAPIIR
  MLIRCAPIRegistration
  ${dialect_libs})
