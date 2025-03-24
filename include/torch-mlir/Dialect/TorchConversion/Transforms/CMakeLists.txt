set(LLVM_TARGET_DEFINITIONS Passes.td)

mlir_tablegen(Passes.h.inc -gen-pass-decls ${TORCH_MLIR_TABLEGEN_FLAGS})

add_public_tablegen_target(TorchMLIRTorchConversionPassIncGen)

add_mlir_doc(Passes TorchMLIRTorchConversionTransforms ./ -gen-pass-doc)
