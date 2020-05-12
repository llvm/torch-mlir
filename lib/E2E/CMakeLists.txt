add_mlir_library(NPCOMPE2E
  E2E.cpp
  LowerRankedShapes.cpp
  LowerToHybridTensorMemRef.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SRC_DIR}/include/npcomp/E2E

  DEPENDS
  NPCOMPE2EPassIncGen
  MLIRLinalgOps

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLinalgOps
  MLIRStandardOps
  )
