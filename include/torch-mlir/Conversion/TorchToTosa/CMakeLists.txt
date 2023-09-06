set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls)
add_public_tablegen_target(TorchMLIRConversionTosaPassIncGen)

add_mlir_doc(Passes TorchMLIRConversionTosaPasses ./ -gen-pass-doc)
