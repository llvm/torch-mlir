add_npcomp_conversion_library(NPCOMPATenPasses
  Passes.cpp
  ATenLayerNamePass.cpp
  ATenOpReport.cpp
  LivenessReport.cpp
  RecognizeKernelsPass.cpp
  ReturnEliminationPass.cpp

  DEPENDS
  MLIRATenToStdIncGen
  NPCOMPATenPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
)
