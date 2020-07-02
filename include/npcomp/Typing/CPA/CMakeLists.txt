set(LLVM_TARGET_DEFINITIONS Interfaces.td)
mlir_tablegen(Interfaces.h.inc -gen-type-interface-decls)
mlir_tablegen(Interfaces.cpp.inc -gen-type-interface-defs)
add_public_tablegen_target(NPCOMPTypingCPAInterfacesIncGen)
