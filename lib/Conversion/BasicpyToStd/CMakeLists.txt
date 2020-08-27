add_mlir_conversion_library(NPCOMPBasicpyToSTD
  Passes.cpp
  PrimitiveOpsConversion.cpp

  DEPENDS
  NPCOMPConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  NPCOMPBasicpyDialect
)
