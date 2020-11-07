add_subdirectory(ATenToTCF)
add_subdirectory(BasicpyToStd)
add_subdirectory(NumpyToTCF)
add_subdirectory(TCFToStd)
add_subdirectory(TCFToTCP)

if(NPCOMP_ENABLE_IREE)
add_subdirectory(BasicpyToIREEVM)
endif()

add_npcomp_library(NPCOMPConversionPasses
  Passes.cpp

  DEPENDS
  NPCOMPConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  NPCOMPBasicpyToSTD
  NPCOMPNumpyToTCF
  NPCOMPTCFToTCP
)
