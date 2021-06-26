add_subdirectory(Runtime)
add_subdirectory(JITHelpers)

add_npcomp_library(NPCOMPRefBackend
  RefBackend.cpp
  LowerToLLVM.cpp
  LowerToRefbackrtABI.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SRC_DIR}/include/npcomp/RefBackend

  DEPENDS
  NPCOMPRefBackendPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLinalg
  MLIRLinalgToLLVM
  MLIRSCFToStandard
  MLIRSCFTransforms
  MLIRShapeToStandard
  MLIRStandard
  MLIRStandardOpsTransforms
  MLIRStandardToLLVM
  )

mlir_check_all_link_libraries(NPCOMPRefBackend)
