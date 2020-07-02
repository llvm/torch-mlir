add_library(NPCOMPTypingCPA
  Interfaces.cpp
  Support.cpp
)

target_link_libraries(NPCOMPTypingCPA
  MLIRIR
)

add_dependencies(NPCOMPTypingCPA
  NPCOMPTypingCPAInterfacesIncGen
)
