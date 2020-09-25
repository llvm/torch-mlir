add_mlir_library(NPCOMPE2E
  BypassShapes.cpp
  E2E.cpp
  LowerShapeConstraints.cpp
  LowerToLLVM.cpp
  LowerToNpcomprtABI.cpp
  TensorToMemref/LowerConstantTensorsToMemref.cpp
  TensorToMemref/LowerShapedResultsToMemref.cpp
  TensorToMemref/LowerStdToMemref.cpp
  TensorToMemref/LowerStructuralToMemref.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SRC_DIR}/include/npcomp/E2E

  DEPENDS
  NPCOMPE2EPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLinalgOps
  MLIRSCFToStandard
  MLIRShapeToStandard
  MLIRStandardOps
  MLIRStandardOpsTransforms
  MLIRStandardToLLVM
  )

mlir_check_all_link_libraries(NPCOMPE2E)
