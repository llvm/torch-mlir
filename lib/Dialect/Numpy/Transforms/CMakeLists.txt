add_mlir_conversion_library(NPCOMPNumpyPasses
  PublicFunctionToTensor.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/npcomp/Dialect/Numpy/Transforms

  DEPENDS
  NPCOMPNumpyPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  NPCOMPNumpyDialect
)
