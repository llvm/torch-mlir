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
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

#-------------------------------------------------------------------------------
# Options and settings
#-------------------------------------------------------------------------------

option(NPCOMP_ENABLE_IREE "Enables the IREE backend (must configure location via IREE_DIR)." OFF)
option(NPCOMP_ENABLE_REFJIT "Enables the reference JIT backend." ON)
option(NPCOMP_BUILD_NPCOMP_DYLIB "Enables shared build of NPCOMP dylib (depends on LLVM/MLIR dylib support)" ON)
set(NPCOMP_IREE_SRCDIR "" CACHE STRING "If building IREE, then setting this elects to build from a source directory (versus installed package)")

# Turn on -gsplit-dwarf if requested in debug builds.
if (NPCOMP_USE_SPLIT_DWARF AND
    ((CMAKE_BUILD_TYPE STREQUAL "Debug") OR
     (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")))
  # Limit to clang and gcc so far. Add compilers supporting this option.
  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR
      CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-gsplit-dwarf)
  endif()
endif()

#-------------------------------------------------------------------------------
# MSVC defaults
#-------------------------------------------------------------------------------

if(MSVC)
    add_compile_options(
        $<$<CONFIG:>:/MD>
        $<$<CONFIG:Debug>:/MD>
        $<$<CONFIG:Release>:/MD>
    )
endif()

#-------------------------------------------------------------------------------
# MLIR/LLVM Configuration
#-------------------------------------------------------------------------------

find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

#set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)

# Define the default arguments to use with 'lit', and an option for the user to
# override.
set(LIT_ARGS_DEFAULT "-sv")
if (MSVC_IDE OR XCODE)
  set(LIT_ARGS_DEFAULT "${LIT_ARGS_DEFAULT} --no-progress-bar")
endif()
set(LLVM_LIT_ARGS "${LIT_ARGS_DEFAULT}" CACHE STRING "Default options for lit")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(AddNPCOMP)
include(HandleLLVMOptions)
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})
set(NPCOMP_TABLEGEN_ARGS "")

#-------------------------------------------------------------------------------
# Optional feature selection
#-------------------------------------------------------------------------------

if(NPCOMP_ENABLE_REFJIT)
  add_compile_definitions(NPCOMP_ENABLE_REFJIT)
  message(STATUS "Reference JIT backend enabled")
endif()

#-------------------------------------------------------------------------------
# IREE configuration
#-------------------------------------------------------------------------------

function(npcomp_add_iree_src)
  # TODO: Find a way to fix this upstream as globally demoting warnings
  # like this is a really bad thing to be doing. Also, Abseil seems to try
  # really hard to keep us from touching these so some still get through.
  add_compile_options(-Wno-sign-compare -Wno-unused-template)
  add_subdirectory("${NPCOMP_IREE_SRCDIR}" "iree" EXCLUDE_FROM_ALL)
endfunction()

if(NPCOMP_ENABLE_IREE)
  add_compile_definitions(NPCOMP_ENABLE_IREE)
  string(APPEND NPCOMP_TABLEGEN_ARGS "-DNPCOMP_ENABLE_IREE")
  if(NPCOMP_IREE_SRCDIR)
    message(STATUS "Depending on IREE source: ${NPCOMP_IREE_SRCDIR}")
    set(IREE_BUILD_TESTS OFF CACHE BOOL "Override IREE setting" FORCE)
    set(IREE_BUILD_SAMPLES OFF CACHE BOOL "Override IREE setting" FORCE)
    set(IREE_BUILD_PYTHON_BINDINGS ON CACHE BOOL "Override IREE setting" FORCE)
    set(IREE_MLIR_DEP_MODE "DISABLED" CACHE STRING "Override IREE setting")
    npcomp_add_iree_src()
  else()
    find_package(IREE REQUIRED CONFIG)
  endif()

  message(STATUS "IREE INCLUDE DIRS: ${IREE_PUBLIC_INCLUDE_DIRS}")
  include_directories("${IREE_PUBLIC_INCLUDE_DIRS}")
endif()

#-------------------------------------------------------------------------------
# Python Configuration
#-------------------------------------------------------------------------------

set(NPCOMP_PYTHON_BINDINGS_VERSION_LOCKED 1 CACHE BOOL
       "Links to specific python libraries, resolving all symbols.")

# TODO(laurenzo): Rationalize with how this is done elsewhere
find_package(PythonInterp REQUIRED)
find_package(PythonLibs)
message(STATUS "Found python include dirs: ${PYTHON_INCLUDE_DIRS}")
message(STATUS "Found ppython libraries: ${PYTHON_LIBRARIES}")
find_package(pybind11 CONFIG REQUIRED)
message(STATUS "Found pybind11 v${pybind11_VERSION}: ${pybind11_INCLUDE_DIRS}")

#-------------------------------------------------------------------------------
# Pytorch Configuration
#-------------------------------------------------------------------------------

find_package(Torch)

#-------------------------------------------------------------------------------
# Directory setup
#-------------------------------------------------------------------------------

set(MLIR_NPCOMP_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(MLIR_NPCOMP_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

add_custom_target(check-npcomp)
add_custom_target(check-all)
add_dependencies(check-all check-npcomp)

add_subdirectory(include/npcomp)
add_subdirectory(lib)
add_subdirectory(python)
add_subdirectory(test)
add_subdirectory(frontends)

# Tools needs to come late to ensure that NPCOMP_ALL_LIBS is populated.
# Generally things after this point may depend on NPCOMP_ALL_LIBS or libNPCOMP.so.
add_subdirectory(tools)
