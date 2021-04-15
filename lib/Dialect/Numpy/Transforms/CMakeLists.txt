add_npcomp_conversion_library(NPCOMPNumpyPasses
  ArrayToTensor.cpp
  Passes.cpp
  PublicFunctionToTensor.cpp
  RefinePublicReturn.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/npcomp/Dialect/Numpy/Transforms

  DEPENDS
  NPCOMPNumpyPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransformUtils
  NPCOMPNumpyDialect
)
