set(LLVM_TARGET_DEFINITIONS ATenToStd.td)
mlir_tablegen(ATenToStd.cpp.inc -gen-rewriters)
add_public_tablegen_target(MLIRATenToStdIncGen)
