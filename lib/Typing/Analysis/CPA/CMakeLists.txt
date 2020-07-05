add_library(NPCOMPTypingCPA
  Algorithm.cpp
  Interfaces.cpp
  Types.cpp
)

target_link_libraries(NPCOMPTypingCPA
  MLIRIR
)

add_dependencies(NPCOMPTypingCPA
  NPCOMPTypingCPAInterfacesIncGen
)
