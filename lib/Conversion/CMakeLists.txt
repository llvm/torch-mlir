add_subdirectory(BasicpyToStd)
add_subdirectory(NumpyToTCF)
add_subdirectory(TCFToTCP)
add_subdirectory(TCPToLinalg)

if(NPCOMP_ENABLE_IREE)
add_subdirectory(BasicpyToIREEVM)
endif()

add_mlir_library(NPCOMPConversionPasses
  Passes.cpp

  DEPENDS
  NPCOMPConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  NPCOMPBasicpyToSTD
  NPCOMPNumpyToTCF
  NPCOMPTCFToTCP
  NPCOMPTCPToLinalg
)
