add_subdirectory(ATenToLinalg)
add_subdirectory(ATenToTCF)
add_subdirectory(BasicpyToStd)
add_subdirectory(NumpyToTCF)
add_subdirectory(TCFToLinalg)
add_subdirectory(TCFToStd)
add_subdirectory(TCFToTCP)

add_npcomp_library(NPCOMPConversionPasses
  Passes.cpp

  DEPENDS
  NPCOMPConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  NPCOMPATenToTCF
  NPCOMPBasicpyToSTD
  NPCOMPNumpyToTCF
  NPCOMPTCFToLinalg
  NPCOMPTCFToStd
  NPCOMPTCFToTCP
)
