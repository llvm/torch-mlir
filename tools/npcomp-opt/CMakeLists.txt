get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        # Note: the following must also match python_native/CMakeLists.txt
        NPCOMPE2E
        NPCOMPTCP
        NPCOMPTCF
        NPCOMPNpcompRt
        NPCOMPNumpyDialect
        NPCOMPBasicpyDialect
        )
add_llvm_executable(npcomp-opt npcomp-opt.cpp)

llvm_update_compile_flags(npcomp-opt)
target_link_libraries(npcomp-opt PRIVATE ${LIBS})
