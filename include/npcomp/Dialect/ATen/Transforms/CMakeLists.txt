# ------------------------------------------------------------------------------
# Declarative pass declarations
# ------------------------------------------------------------------------------

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls)
add_public_tablegen_target(NPCOMPATenPassIncGen)

# ------------------------------------------------------------------------------
# DRR
# ------------------------------------------------------------------------------

set(LLVM_TARGET_DEFINITIONS ATenToStd.td)
mlir_tablegen(ATenToStd.cpp.inc -gen-rewriters)
add_public_tablegen_target(MLIRATenToStdIncGen)
