add_library(NPCOMPTypingCPASupport
  CPAIrHelpers.cpp
)

target_link_libraries(NPCOMPTypingCPASupport
  MLIRIR
  NPCOMPTypingCPA
  NPCOMPBasicpyDialect
)
