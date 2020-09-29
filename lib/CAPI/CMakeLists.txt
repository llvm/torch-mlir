add_mlir_library(NPCOMPCAPIRegistration
  Registration.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  NPCOMPInitAll
  )
