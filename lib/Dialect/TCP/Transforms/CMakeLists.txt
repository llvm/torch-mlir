add_npcomp_conversion_library(NPCOMPTCPPasses
  Passes.cpp
  Bufferize.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/npcomp/Dialect/TCP/Transforms

  DEPENDS
  NPCOMPTCPPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  NPCOMPRefbackDialect
  NPCOMPTCPDialect
)
