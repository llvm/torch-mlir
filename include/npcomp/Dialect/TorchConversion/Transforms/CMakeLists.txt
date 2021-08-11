set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls)
add_public_tablegen_target(NPCOMPTorchConversionPassIncGen)

add_mlir_doc(Passes NPCOMPTorchConversionTransforms ./ -gen-pass-doc)
