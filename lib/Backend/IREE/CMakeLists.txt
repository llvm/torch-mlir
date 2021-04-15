add_npcomp_library(NPCOMPIREEBackend
  LowerLinkage.cpp
  Passes.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SRC_DIR}/include/npcomp/Backend/IREE

  DEPENDS
  NPCOMPIREEBackendPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  )

mlir_check_all_link_libraries(NPCOMPIREEBackend)
