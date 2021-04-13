add_subdirectory(Common)

if(NPCOMP_ENABLE_REFJIT)
  add_subdirectory(RefJIT)
endif()

# Currently this doesn't introduce any actual dependency on IREE, so add it
# unconditionally.
# TODO: Put this behind the NPCOMP_ENABLE_IREE flag.
add_subdirectory(IREE)

