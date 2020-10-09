add_npcomp_library(NPCOMPTypingCPA
  Algorithm.cpp
  Interfaces.cpp
  Types.cpp

  DEPENDS
  NPCOMPTypingCPAInterfacesIncGen

  LINK_LIBS
  PUBLIC
  MLIRIR
)
