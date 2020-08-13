add_mlir_dialect_library(NPCOMPATenDialect
  ATenDialect.cpp
  ATenDialectOpStats.cpp
  ATenPasses.cpp
  ATenLayerNamePass.cpp
  ATenLoweringPass.cpp
  ATenOpReport.cpp
  ATenToStd.cpp
  LivenessReport.cpp
  ReturnEliminationPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/dialect/include
  ${PROJECT_BINARY_DIR}/dialect/include

  DEPENDS
  MLIRATenIncGen
  MLIRATenEnumsIncGen
  MLIRATenOpInterfacesIncGen
  MLIRATenToStdIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRTransformUtils
  )
