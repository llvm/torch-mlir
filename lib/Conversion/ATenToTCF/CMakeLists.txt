add_npcomp_conversion_library(NPCOMPATenToTCF
  ConvertATenToTCFPass.cpp
  CoreOpConversionPatterns.cpp

  DEPENDS
  NPCOMPConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  NPCOMPATenDialect
  NPCOMPTCFDialect
)
