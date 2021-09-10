################################################################################
# Manage python source files
# Collapse all local python sources to the project level python/ directory.
################################################################################

include(NpcompPython)

npcomp_python_create_symlinks(
  ${MLIR_NPCOMP_PYTHON_PACKAGES_DIR}/npcomp_torch ${CMAKE_CURRENT_SOURCE_DIR})
