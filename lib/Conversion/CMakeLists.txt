add_subdirectory(BasicpyToStd)
add_subdirectory(TCFToTCP)
add_subdirectory(TCPToLinalg)

if(NPCOMP_ENABLE_IREE)
add_subdirectory(BasicpyToIREEVM)
endif()
