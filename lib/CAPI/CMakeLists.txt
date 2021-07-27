set(LLVM_LINK_COMPONENTS
  Core
  Support
  nativecodegen
  )

add_npcomp_library(NPCOMPCAPI
  InitLLVM.cpp
  RefJITBackend.cpp
  Registration.cpp
  BasicpyTypes.cpp
  NumpyTypes.cpp
  TorchTypes.cpp

  LINK_LIBS PUBLIC
  MLIRExecutionEngine
  MLIRLLVMIR
  MLIRTargetLLVMIRExport
  NPCOMPInitAll
  NPCOMPBasicpyDialect
  NPCOMPNumpyDialect
  NPCOMPRefBackendJITHelpers
  NPCOMPRuntime
  NPCOMPTorchDialect

  # MLIR CAPI deps
  MLIRCAPIIR
  MLIRCAPIRegistration  # TODO: Remove
)
