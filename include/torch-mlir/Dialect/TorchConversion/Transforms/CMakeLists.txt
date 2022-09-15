set(LLVM_TARGET_DEFINITIONS Passes.td)
if(TORCH_MLIR_ENABLE_MHLO)
  mlir_tablegen(Passes.h.inc -gen-pass-decls -DTORCH_MLIR_ENABLE_MHLO)
else()
  mlir_tablegen(Passes.h.inc -gen-pass-decls)
endif()
add_public_tablegen_target(TorchMLIRTorchConversionPassIncGen)

add_mlir_doc(Passes TorchMLIRTorchConversionTransforms ./ -gen-pass-doc)
