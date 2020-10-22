add_npcomp_conversion_library(NPCOMPATenPasses
  ATenPasses.cpp
  ATenLayerNamePass.cpp
  ATenLoweringPass.cpp
  ATenOpReport.cpp
  ATenToStd.cpp
  LivenessReport.cpp
  ReturnEliminationPass.cpp

  DEPENDS
  MLIRATenToStdIncGen
)
