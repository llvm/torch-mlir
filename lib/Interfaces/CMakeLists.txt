add_npcomp_library(NPCOMPInterfaces
  Traits.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SRC_DIR}/include/npcomp/Interfaces

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRStandard
  )

mlir_check_all_link_libraries(NPCOMPInterfaces)
