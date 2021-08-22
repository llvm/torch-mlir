add_npcomp_conversion_library(NPCOMPTorchConversionPasses
  BackendTypeConversion.cpp
  Passes.cpp
  TmpDeleteDeadIREELists.cpp
  VerifyInvariantsBeforeBackendLowering.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/npcomp/Dialect/TorchConversion/Transforms

  DEPENDS
  NPCOMPTorchConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRStandardOpsTransforms
  NPCOMPTorchConversionDialect
  NPCOMPTorchDialect
  NPCOMPTorchPasses
  NPCOMPTorchToIREE
  NPCOMPTorchToLinalg
  NPCOMPTorchToStd
  NPCOMPTorchToSCF
  NPCOMPInterfaces
  MLIRMemRefTransforms
)
