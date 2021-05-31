add_npcomp_conversion_library(NPCOMPTorchToLinalg
  TorchToLinalg.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/npcomp/Conversion/TorchToLinalg

  DEPENDS
  NPCOMPConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRLinalg
  MLIRMath
  MLIRTransformUtils
  NPCOMPTorchDialect
)
