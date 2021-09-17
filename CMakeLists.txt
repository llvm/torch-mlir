cmake_minimum_required(VERSION 3.13.4)

# CMP0116: Ninja generators transform `DEPFILE`s from `add_custom_command()`
# New in CMake 3.20. https://cmake.org/cmake/help/latest/policy/CMP0116.html
if(POLICY CMP0116)
  cmake_policy(SET CMP0116 OLD)
endif()
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
# Options and settings
#-------------------------------------------------------------------------------
set(NPCOMP_MINIMUM_PYTHON_VERSION 3.6)
option(NPCOMP_ENABLE_REFJIT "Enables the reference JIT backend." ON)
set(NPCOMP_IREE_BUILDDIR "../iree-build" CACHE STRING "If building IREE, then setting this elects to build from a source directory (versus installed package)")

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
# Directory setup
#-------------------------------------------------------------------------------

set(MLIR_NPCOMP_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(MLIR_NPCOMP_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(MLIR_NPCOMP_PYTHON_PACKAGES_DIR ${CMAKE_CURRENT_BINARY_DIR}/python_packages)
set(TORCH_MLIR_PYTHON_PACKAGES_DIR "${MLIR_NPCOMP_PYTHON_PACKAGES_DIR}")

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  # If we are not building as a part of LLVM, build NPCOMP as a standalone
  # project, including LLVM as a subdirectory. This gives us the most control
  # and is used for standalone releases.

  #-------------------------------------------------------------------------------
  # Project setup and globals
  #-------------------------------------------------------------------------------

  project(npcomp LANGUAGES CXX C)
  set(CMAKE_C_STANDARD 11)
  set(CMAKE_CXX_STANDARD 14)

  #-------------------------------------------------------------------------------
  # Default and required options.
  #-------------------------------------------------------------------------------

  # CMake library generation settings.
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "We are actually building a static mondo-lib")
  set(CMAKE_PLATFORM_NO_VERSIONED_SONAME ON CACHE BOOL "Python soname linked libraries are bad")
  set(CMAKE_VISIBILITY_INLINES_HIDDEN ON CACHE BOOL "Hide inlines")
  set(CMAKE_CXX_VISIBILITY_PRESET hidden)

  # Improved LLVM defaults for a standalone build.
  set(LLVM_ENABLE_PROJECTS mlir CACHE STRING "LLVM projects")
  set(LLVM_ENABLE_Z3_SOLVER OFF CACHE BOOL "Disable Z3")
  set(LLVM_ENABLE_ZLIB OFF CACHE BOOL "Disable ZLIB")
  set(LLVM_TARGETS_TO_BUILD "host" CACHE STRING "Only build for the host")
  set(LLVM_INCLUDE_EXAMPLES OFF CACHE BOOL "Disable examples")
  # TODO: MLIR is a "tool"
  set(LLVM_INCLUDE_TOOLS ON CACHE BOOL "Disable tools")
  set(LLVM_INCLUDE_TESTS ON CACHE BOOL "Disable tests")
  set(MLIR_BINDINGS_PYTHON_LOCK_VERSION ON CACHE BOOL "Link against libpython for development (should be disabled for production)")

  # Required LLVM settings.
  set(MLIR_ENABLE_BINDINGS_PYTHON ON CACHE BOOL "Enable MLIR python bindings" FORCE)

  #-------------------------------------------------------------------------------
  # MLIR/LLVM Build Setup
  # TODO: It would be nice to have a better setup than this for sub including
  # MLIR.
  #-------------------------------------------------------------------------------
  if(NOT LLVM_MAIN_SRC_DIR)
    set(LLVM_MAIN_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/llvm-project/llvm")
  endif()
  if(NOT MLIR_MAIN_SRC_DIR)
    set(MLIR_MAIN_SRC_DIR "${LLVM_MAIN_SRC_DIR}/../mlir")
  endif()

  set(LLVM_MAIN_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/llvm")
  set(MLIR_MAIN_BINARY_DIR "${LLVM_MAIN_BINARY_DIR}/tools/mlir")

  set(LLVM_INCLUDE_DIR "${LLVM_MAIN_SRC_DIR}/include")
  set(LLVM_GENERATED_INCLUDE_DIR "${LLVM_MAIN_BINARY_DIR}/include")
  set(MLIR_INCLUDE_DIR "${MLIR_MAIN_SRC_DIR}/include")
  set(MLIR_GENERATED_INCLUDE_DIR "${MLIR_MAIN_BINARY_DIR}/include")
  set(MLIR_TABLEGEN_EXE "$<TARGET_FILE:mlir-tblgen>")
  include_directories(SYSTEM "${LLVM_INCLUDE_DIR}")
  include_directories(SYSTEM "${LLVM_GENERATED_INCLUDE_DIR}")
  include_directories(SYSTEM "${MLIR_INCLUDE_DIR}")
  include_directories(SYSTEM "${MLIR_GENERATED_INCLUDE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${MLIR_MAIN_SRC_DIR}/cmake/modules")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_MAIN_SRC_DIR}/cmake/modules")

  # Pre-configure the Python environment using the MLIR macros so that they
  # are in scope and subsequent include of LLVM will match them.
  include(MLIRDetectPythonEnv)
  find_package(Python3 ${NPCOMP_MINIMUM_PYTHON_VERSION}
    COMPONENTS Interpreter Development NumPy REQUIRED)
  message(STATUS "Found python include dirs: ${Python3_INCLUDE_DIRS}")
  message(STATUS "Found python libraries: ${Python3_LIBRARIES}")
  mlir_detect_pybind11_install()
  find_package(pybind11 2.6 CONFIG REQUIRED)
  message(STATUS "Found pybind11 v${pybind11_VERSION}: ${pybind11_INCLUDE_DIR}")
  message(STATUS "Python prefix = '${PYTHON_MODULE_PREFIX}', "
                 "suffix = '${PYTHON_MODULE_SUFFIX}', "
                 "extension = '${PYTHON_MODULE_EXTENSION}")

  # Include LLVM_EXTERNAL_PROJECTS.
  set(LLVM_EXTERNAL_PROJECTS "iree-dialects;torch-mlir")
  set(LLVM_EXTERNAL_IREE_DIALECTS_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/iree-dialects")
  set(LLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/torch-mlir")

  # LLVM configuration.
  message(STATUS "*** ADDING LLVM ***")
  add_subdirectory(
    "${CMAKE_CURRENT_SOURCE_DIR}/external/llvm-project/llvm"
    "${LLVM_MAIN_BINARY_DIR}"
    EXCLUDE_FROM_ALL)
  message(STATUS "*** LLVM DONE ***")

  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)

  set(LLVM_EXTERNAL_LIT "${LLVM_MAIN_BINARY_DIR}/bin/llvm-lit")
  set(LLVM_TOOLS_DIR "${LLVM_MAIN_BINARY_DIR}/bin")

  # Define the default arguments to use with 'lit', and an option for the user to
  # override.
  set(LIT_ARGS_DEFAULT "-sv")
  if (MSVC_IDE OR XCODE)
    set(LIT_ARGS_DEFAULT "${LIT_ARGS_DEFAULT} --no-progress-bar")
  endif()
  set(LLVM_LIT_ARGS "${LIT_ARGS_DEFAULT}" CACHE STRING "Default options for lit")

  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  include(AddMLIRPython)
  include(HandleLLVMOptions)
  set(NPCOMP_BUILT_STANDALONE 1)
else()
  # TODO: RE-ENABLE EXTERNAL UNIFIED BUILD
  message(FATAL_ERROR "External project builds of npcomp are currently not available")
  # Otherwise, we are building as a part of LLVM, and we need to set up some
  # variables and includes.
  # set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir/include ) # --src-root
  # set(MLIR_INCLUDE_DIR ${LLVM_MAIN_SRC_DIR}/../mlir/include ) # --includedir
  # set(MLIR_TABLEGEN_OUTPUT_DIR ${CMAKE_BINARY_DIR}/tools/mlir/include)
  # set(MLIR_TABLEGEN_EXE $<TARGET_FILE:mlir-tblgen>)
  # include_directories(SYSTEM ${MLIR_INCLUDE_DIR})
  # include_directories(SYSTEM ${MLIR_TABLEGEN_OUTPUT_DIR})

  # set(BACKEND_PACKAGE_STRING "${PACKAGE_STRING}")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")
include(AddNPCOMP)
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/iree-dialects/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/llvm/tools/iree-dialects/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/torch-mlir/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/llvm/tools/torch-mlir/include)
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
# Subdirectories and aggregate testing targets.
#-------------------------------------------------------------------------------

add_custom_target(check-npcomp)
add_custom_target(check-npcomp-all)
add_dependencies(check-npcomp-all
  check-npcomp
  check-npcomp-python
  check-torch-mlir
  check-torch-mlir-plugin
  )

add_subdirectory(include/npcomp)
add_subdirectory(lib)
add_subdirectory(python)
add_subdirectory(test)
add_subdirectory(tools)
