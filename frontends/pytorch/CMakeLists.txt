#-------------------------------------------------------------------------------
# Options and settings
#-------------------------------------------------------------------------------

option(NPCOMP_ENABLE_TORCH_TYPE_DISPATCH "Enables the legacy ATen Type dispatch code path" OFF)
if(NPCOMP_ENABLE_TORCH_TYPE_DISPATCH)
  add_compile_definitions(NPCOMP_ENABLE_TORCH_TYPE_DISPATCH)
  message(STATUS "Legacy Torch type dispatch mechanism enabled")
endif()

#-------------------------------------------------------------------------------
# Subdirectories
#-------------------------------------------------------------------------------

# TODO: This sub-directory does not need to be gated on the type dispatch
# mechanism, but it presently has some dependencies on an older pytorch version
# and is being excluded until those can be resolved.
if(NPCOMP_ENABLE_TORCH_TYPE_DISPATCH)
  add_subdirectory(lib)
endif()

add_subdirectory(csrc)
add_subdirectory(test)
