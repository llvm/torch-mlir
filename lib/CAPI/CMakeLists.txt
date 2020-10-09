set(LLVM_LINK_COMPONENTS
  Core
  Support
  nativecodegen
  )

add_npcomp_library(NPCOMPCAPI
  InitLLVM.cpp
  Registration.cpp
  Types.cpp

  LINK_LIBS PUBLIC
  MLIRExecutionEngine
  MLIRLLVMIR
  MLIRTargetLLVMIR
  NPCOMPInitAll
  NPCOMPBasicpyDialect
  NPCOMPNumpyDialect
  )
