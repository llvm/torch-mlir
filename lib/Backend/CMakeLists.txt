add_subdirectory(Common)

# Currently this doesn't introduce any actual dependency on IREE, so add it
# unconditionally.
# TODO: Put this behind the NPCOMP_ENABLE_IREE flag.
add_subdirectory(IREE)

