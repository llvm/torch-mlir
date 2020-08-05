set(LLVM_LINK_COMPONENTS
  Core
  Support
  nativecodegen
  )

add_llvm_tool(npcomp-run-mlir
  npcomp-run-mlir.cpp
  )
llvm_update_compile_flags(npcomp-run-mlir)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
target_link_libraries(npcomp-run-mlir PRIVATE
  MLIRAnalysis
  MLIREDSC
  MLIRExecutionEngine
  MLIRIR
  MLIRJitRunner
  MLIRLLVMIR
  MLIRParser
  MLIRTargetLLVMIR
  MLIRSupport
  NPCOMPInitAll
  NPCOMPJITRuntime
  ${conversion_libs}
  ${dialect_libs}
  )
add_dependencies(npcomp-run-mlir
  NPCOMPCompilerRuntimeShlib
  )
