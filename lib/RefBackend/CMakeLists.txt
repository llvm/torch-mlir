add_subdirectory(Runtime)
add_subdirectory(JITHelpers)

add_npcomp_library(NPCOMPRefBackend
  RefBackend.cpp
  LowerToLLVM.cpp
  LowerToRefbackrtABI.cpp
  TensorToMemref/LowerConstantTensorsToMemref.cpp
  TensorToMemref/LowerStdToMemref.cpp
  TensorToMemref/LowerStructuralToMemref.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SRC_DIR}/include/npcomp/RefBackend

  DEPENDS
  NPCOMPRefBackendPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLinalg
  MLIRSCFToStandard
  MLIRShapeToStandard
  MLIRStandard
  MLIRStandardOpsTransforms
  MLIRStandardToLLVM
  )

mlir_check_all_link_libraries(NPCOMPRefBackend)
