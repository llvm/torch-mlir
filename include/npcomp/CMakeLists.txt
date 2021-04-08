# Currently this doesn't introduce any actual dependency on IREE, so add it
# unconditionally.
# TODO: Put this behind the NPCOMP_ENABLE_IREE flag.
add_subdirectory(Backend/IREE)

add_subdirectory(Conversion)
add_subdirectory(Dialect)
add_subdirectory(RefBackend)
add_subdirectory(Typing)
