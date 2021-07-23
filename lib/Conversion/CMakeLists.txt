add_subdirectory(TorchToLinalg)
add_subdirectory(TorchToSCF)
add_subdirectory(TorchToStd)
add_subdirectory(BasicpyToStd)
add_subdirectory(NumpyToTCF)
add_subdirectory(TCFToLinalg)
add_subdirectory(TCFToStd)
add_subdirectory(TCFToTCP)

get_property(npcomp_conversion_libs GLOBAL PROPERTY NPCOMP_CONVERSION_LIBS)

add_npcomp_library(NPCOMPConversionPasses
  Passes.cpp

  DEPENDS
  NPCOMPConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  ${npcomp_conversion_libs}
)
