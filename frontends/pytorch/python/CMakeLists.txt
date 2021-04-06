################################################################################
# Manage python source files
# Collapse all local python sources to the project level python/ directory.
################################################################################

npcomp_python_create_symlinks(${CMAKE_BINARY_DIR}/python ${CMAKE_CURRENT_SOURCE_DIR})

install(DIRECTORY . DESTINATION python COMPONENT python)
