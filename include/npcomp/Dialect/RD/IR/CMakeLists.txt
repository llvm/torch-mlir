add_mlir_dialect(RDOps rd)

# TODO: convert to using `add_mlir_interface`.
set(LLVM_TARGET_DEFINITIONS RDDatasetInterface.td)
mlir_tablegen(RDDatasetInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(RDDatasetInterface.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRRDDatasetInterfaceIncGen)
add_dependencies(mlir-generic-headers MLIRRDDatasetInterfaceIncGen)
# </convert>
