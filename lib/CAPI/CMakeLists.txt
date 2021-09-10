set(LLVM_LINK_COMPONENTS
  Core
  Support
  nativecodegen
  )

add_npcomp_library(NPCOMPCAPI
  InitLLVM.cpp
  RefJITBackend.cpp
  Registration.cpp

  LINK_LIBS PUBLIC
  MLIRExecutionEngine
  MLIRLLVMIR
  MLIRTargetLLVMIRExport
  NPCOMPInitAll
  NPCOMPRefBackendJITHelpers
  NPCOMPRuntime
  TorchMLIRTorchDialect
  TorchMLIRInitAll

  # MLIR CAPI deps
  MLIRCAPIIR
  MLIRCAPIRegistration  # TODO: Remove
)
