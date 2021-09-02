add_npcomp_library(NPCOMPCommonBackend
  DeleteDeadIREELists.cpp
  VerifyBackendContract.cpp
  Passes.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SRC_DIR}/include/npcomp/Backend/Common

  DEPENDS
  NPCOMPCommonBackendPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLinalg
  MLIRTensor
  MLIRStandard
  MLIRMath
  IREEDialectsIREEDialect
  )

mlir_check_all_link_libraries(NPCOMPCommonBackend)
