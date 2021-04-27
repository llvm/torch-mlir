set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls ${NPCOMP_TABLEGEN_ARGS})
add_public_tablegen_target(NPCOMPConversionPassIncGen)

add_mlir_doc(Passes NPCOMPConversionPasses ./ -gen-pass-doc)
