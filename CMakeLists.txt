cmake_minimum_required(VERSION 3.10)

if(POLICY CMP0068)
  cmake_policy(SET CMP0068 NEW)
  set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
endif()

if(POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

#-------------------------------------------------------------------------------
# Project setup and globals
#-------------------------------------------------------------------------------

project(npcomp LANGUAGES CXX C)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 14)

#-------------------------------------------------------------------------------
# Options and settings
#-------------------------------------------------------------------------------

option(NPCOMP_ENABLE_IREE "Enables the IREE backend (must configure location via IREE_DIR)." OFF)
set(NPCOMP_IREE_SRCDIR "" CACHE STRING "If building IREE, then setting this elects to build from a source directory (versus installed package)")

#-------------------------------------------------------------------------------
# MLIR/LLVM Configuration
#-------------------------------------------------------------------------------

find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})
set(NPCOMP_TABLEGEN_ARGS "")

#-------------------------------------------------------------------------------
# IREE configuration
#-------------------------------------------------------------------------------

if(NPCOMP_ENABLE_IREE)
  add_compile_definitions(NPCOMP_ENABLE_IREE)
  string(APPEND NPCOMP_TABLEGEN_ARGS "-DNPCOMP_ENABLE_IREE")
  if(NPCOMP_IREE_SRCDIR)
    message(STATUS "Depending on IREE source: ${NPCOMP_IREE_SRCDIR}")
    set(IREE_BUILD_TESTS OFF CACHE BOOL "Override IREE setting")
    set(IREE_BUILD_SAMPLES OFF CACHE BOOL "Override IREE setting")
    set(IREE_MLIR_DEP_MODE "DISABLED" CACHE STRING "Override IREE setting")
    add_subdirectory("${NPCOMP_IREE_SRCDIR}" "iree" EXCLUDE_FROM_ALL)
  else()
    find_package(IREE REQUIRED CONFIG)
  endif()

  message(STATUS "IREE INCLUDE DIRS: ${IREE_PUBLIC_INCLUDE_DIRS}")
  include_directories("${IREE_PUBLIC_INCLUDE_DIRS}")
endif()

#-------------------------------------------------------------------------------
# Python Configuration
#-------------------------------------------------------------------------------

# TODO(laurenzo): Rationalize with how this is done elsewhere
find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)
message(STATUS "Found python include dirs: ${PYTHON_INCLUDE_DIRS}")
message(STATUS "Found ppython libraries: ${PYTHON_LIBRARIES}")
find_package(pybind11 CONFIG REQUIRED)
message(STATUS "Found pybind11 v${pybind11_VERSION}: ${pybind11_INCLUDE_DIRS}")

#-------------------------------------------------------------------------------
# Directory setup
#-------------------------------------------------------------------------------

set(MLIR_NPCOMP_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(MLIR_NPCOMP_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

add_custom_target(check-npcomp)

add_subdirectory(include/npcomp)
add_subdirectory(lib)
add_subdirectory(python)
add_subdirectory(python_native)
add_subdirectory(pytest)
add_subdirectory(runtime)
add_subdirectory(tools)
add_subdirectory(test)
