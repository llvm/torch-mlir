get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        NPCOMPInitAll
)
add_llvm_executable(npcomp-opt npcomp-opt.cpp)

llvm_update_compile_flags(npcomp-opt)
target_link_libraries(npcomp-opt PRIVATE ${LIBS})
