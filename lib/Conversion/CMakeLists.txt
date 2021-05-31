add_subdirectory(TorchToLinalg)
add_subdirectory(TorchToStd)
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
  NPCOMPBasicpyToSTD
  NPCOMPNumpyToTCF
  NPCOMPTCFToLinalg
  NPCOMPTCFToStd
  NPCOMPTCFToTCP
  NPCOMPTorchToLinalg
  NPCOMPTorchToStd
)
