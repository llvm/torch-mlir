add_npcomp_library(NPCOMPTypingCPASupport
  CPAIrHelpers.cpp

  LINK_LIBS
  PUBLIC
  MLIRIR
  NPCOMPTypingCPA
  NPCOMPBasicpyDialect
)
