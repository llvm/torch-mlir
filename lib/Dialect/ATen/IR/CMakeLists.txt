add_npcomp_dialect_library(NPCOMPATenDialect
  ATenDialect.cpp
  ATenDialectOpStats.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/npcomp/Dialect/ATen

  DEPENDS
  MLIRATenIncGen
  #MLIRATenEnumsIncGen
  MLIRATenOpInterfacesIncGen
  #MLIRATenToStdIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRTransformUtils
)
