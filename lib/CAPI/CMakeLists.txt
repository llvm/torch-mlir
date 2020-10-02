add_mlir_library(NPCOMPCAPIIR
  Registration.cpp
  Types.cpp
  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  NPCOMPInitAll
  NPCOMPBasicpyDialect
  NPCOMPNumpyDialect
  )
