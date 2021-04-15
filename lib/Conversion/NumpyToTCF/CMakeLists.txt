add_npcomp_conversion_library(NPCOMPNumpyToTCF
  Passes.cpp

  DEPENDS
  NPCOMPConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  NPCOMPBasicpyDialect
  NPCOMPNumpyDialect
  NPCOMPTCFDialect
)
