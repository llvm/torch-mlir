set(LLVM_TARGET_DEFINITIONS TestTorchDialectExtension.td)
mlir_tablegen(TestTorchDialectExtension.h.inc -gen-op-decls)
mlir_tablegen(TestTorchDialectExtension.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRTestTorchDialectExtensionIncGen)

add_mlir_library(MLIRTestTorchDialectExtension
  TestTorchDialectExtension.cpp

  EXCLUDE_FROM_LIBMLIR

  ADDITIONAL_HEADER_DIRS
  ${TORCH_MLIR_BINARY_DIR}/test/TorchExtension

  DEPENDS
  MLIRTestTorchDialectExtensionIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  TorchMLIRTorchDialect
)
