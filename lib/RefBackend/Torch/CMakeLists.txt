add_npcomp_library(NPCOMPRefBackendTorch
  RecognizeTorchFallbackPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SRC_DIR}/include/npcomp/RefBackend/Torch

  DEPENDS
  NPCOMPRefBackendPassIncGen

  LINK_LIBS PUBLIC
  NPCOMPTorchDialect
  )

mlir_check_all_link_libraries(NPCOMPRefBackendTorch)
