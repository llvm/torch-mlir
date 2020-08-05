get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        MLIROptLib
        NPCOMPInitAll
        ${conversion_libs}
        ${dialect_libs}
)
add_llvm_executable(npcomp-opt npcomp-opt.cpp)

llvm_update_compile_flags(npcomp-opt)
target_link_libraries(npcomp-opt PRIVATE ${LIBS})
