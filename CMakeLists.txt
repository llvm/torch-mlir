# MLIR npcomp project.
set(MLIR_NPCOMP_MAIN_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}) # --src-root
set(MLIR_NPCOMP_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include) # --includedir

set(MLIR_NPCOMP_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(MLIR_NPCOMP_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

# TODO(laurenzo): Rationalize with how this is done elsewhere
find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)
message(STATUS "Found python include dirs: ${PYTHON_INCLUDE_DIRS}")
message(STATUS "Found ppython libraries: ${PYTHON_LIBRARIES}")
find_package(pybind11 CONFIG REQUIRED)
message(STATUS "Found pybind11 v${pybind11_VERSION}: ${pybind11_INCLUDE_DIRS}")

# TODO(laurenzo): What is the right way to get include directories for
# cross project dependencies?
include_directories(${MLIR_NPCOMP_INCLUDE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/../mlir/include)
include_directories(${CMAKE_BINARY_DIR}/tools/mlir/include)

add_subdirectory(include/npcomp)
add_subdirectory(lib)
add_subdirectory(python)

