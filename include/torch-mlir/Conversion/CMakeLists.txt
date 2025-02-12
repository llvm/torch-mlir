add_subdirectory(TorchOnnxToTorch)

set(LLVM_TARGET_DEFINITIONS Passes.td)



mlir_tablegen(Passes.h.inc -gen-pass-decls ${TORCH_MLIR_TABLEGEN_FLAGS})

add_public_tablegen_target(TorchMLIRConversionPassIncGen)

add_mlir_doc(Passes TorchMLIRConversionPasses ./ -gen-pass-doc)
