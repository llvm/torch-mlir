# Forked and adapted from MLIRDetectPythonEnv.cmake upstream. The eventual
# goal is to use the upstream configuration but until that is ready, just
# move/adapt these facilities. Please try to not diverge.

# Macros and functions related to detecting details of the Python environment.

# Detects a pybind11 package installed in the current python environment
# and sets variables to allow it to be found. This allows pybind11 to be
# installed via pip, which typically yields a much more recent version than
# the OS install, which will be available otherwise.
function(npcomp_detect_pybind11_install)
  if(pybind11_DIR)
    message(STATUS "Using explicit pybind11 cmake directory: ${pybind11_DIR} (-Dpybind11_DIR to change)")
  else()
    message(STATUS "Checking for pybind11 in python path...")
    execute_process(
      COMMAND "${Python3_EXECUTABLE}"
      -c "import pybind11;print(pybind11.get_cmake_dir(), end='')"
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      RESULT_VARIABLE STATUS
      OUTPUT_VARIABLE PACKAGE_DIR
      ERROR_QUIET)
    if(NOT STATUS EQUAL "0")
      message(CHECK_FAIL "not found (install via 'pip install pybind11' or set pybind11_DIR)")
      return()
    endif()
    set(pybind11_DIR "${PACKAGE_DIR}" PARENT_SCOPE)
  endif()
endfunction()
