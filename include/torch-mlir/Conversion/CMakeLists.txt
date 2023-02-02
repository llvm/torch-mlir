set(LLVM_TARGET_DEFINITIONS Passes.td)
if(TORCH_MLIR_ENABLE_STABLEHLO)
  mlir_tablegen(Passes.h.inc -gen-pass-decls -DTORCH_MLIR_ENABLE_STABLEHLO)
else()
  mlir_tablegen(Passes.h.inc -gen-pass-decls)
endif()
add_public_tablegen_target(TorchMLIRConversionPassIncGen)

add_mlir_doc(Passes TorchMLIRConversionPasses ./ -gen-pass-doc)
