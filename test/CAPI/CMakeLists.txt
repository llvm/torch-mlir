add_npcomp_executable(npcomp-capi-ir-test ir.c)
llvm_update_compile_flags(npcomp-capi-ir-test)

target_link_libraries(npcomp-capi-ir-test
  PRIVATE
  NPCOMP)
