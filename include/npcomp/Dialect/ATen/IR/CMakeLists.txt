set(dialect_namespace aten)
set(LLVM_TARGET_DEFINITIONS ATenOps.td)

mlir_tablegen(ATenOps.h.inc -gen-op-decls)
mlir_tablegen(ATenOps.cpp.inc -gen-op-defs)
mlir_tablegen(ATenDialect.h.inc -gen-dialect-decls -dialect=${dialect_namespace})
mlir_tablegen(ATenEnums.h.inc -gen-enum-decls)
mlir_tablegen(ATenEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRATenIncGen)
add_dependencies(mlir-headers MLIRATenIncGen)

set(LLVM_TARGET_DEFINITIONS ATenOpInterface.td)
mlir_tablegen(ATenOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(ATenOpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRATenOpInterfacesIncGen)
add_dependencies(mlir-generic-headers MLIRATenOpInterfacesIncGen)

add_mlir_doc(ATenDialect -gen-dialect-doc ATenDialect ATen/)
add_mlir_doc(ATenOps -gen-op-doc ATenOps ATen/)
