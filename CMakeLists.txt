cmake_minimum_required(VERSION 3.12)

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

if(POLICY CMP0116)
  cmake_policy(SET CMP0116 OLD)
endif()

#-------------------------------------------------------------------------------
# Project setup and globals
#-------------------------------------------------------------------------------

project(torch-mlir LANGUAGES CXX C)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 14)

macro(torch_mlir_add_llvm_external_project name identifier location)
  message(STATUS "Adding LLVM external project ${name} (${identifier}) -> ${location}")
  if(NOT EXISTS "${location}/CMakeLists.txt")
    message(FATAL_ERROR "External project location ${location} is not valid")
  endif()
  list(APPEND LLVM_EXTERNAL_PROJECTS ${name})
  list(REMOVE_DUPLICATES LLVM_EXTERNAL_PROJECTS)
  set(LLVM_EXTERNAL_${identifier}_SOURCE_DIR ${location} CACHE STRING "" FORCE)
  set(LLVM_EXTERNAL_PROJECTS ${LLVM_EXTERNAL_PROJECTS} CACHE STRING "" FORCE)
endmacro()

torch_mlir_add_llvm_external_project(
  torch-mlir-dialects
  TORCH_MLIR_DIALECTS
  ${CMAKE_CURRENT_SOURCE_DIR}/externals/llvm-external-projects/torch-mlir-dialects)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  # Out-of-tree build

  #-------------------------------------------------------------------------------
  # MLIR/LLVM Configuration
  #-------------------------------------------------------------------------------

  find_package(MLIR REQUIRED CONFIG)
  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)

  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

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
  include(HandleLLVMOptions)

  # Don't try to compile the python extensions at the moment. We need
  # to import lots of dependencies from AddMLIRPython to make this work.
  set(MLIR_ENABLE_BINDINGS_PYTHON 1)
  option(TORCH_MLIR_ENABLE_JIT_IR_IMPORTER "Enables JIT IR Importer" ON)

  set(TORCH-MLIR_BUILT_STANDALONE 1)
  set(BACKEND_PACKAGE_STRING "LLVM ${LLVM_PACKAGE_VERSION}")
  add_subdirectory(externals/llvm-external-projects/torch-mlir-dialects)
else()
  # In-tree build with LLVM_EXTERNAL_PROJECTS=torch-mlir
  # FIXME: This should really be inherited from the LLVM tree.  In particular,
  # it's going to change when cross-compiling.
  set(MLIR_TABLEGEN_EXE mlir-tblgen)

  option(MLIR_ENABLE_BINDINGS_PYTHON "Enables MLIR Python Bindings" OFF)
  option(TORCH_MLIR_ENABLE_JIT_IR_IMPORTER "Enables JIT IR Importer" ON)

  # TODO: Fix this upstream so that global include directories are not needed.
  set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir)
  set(MLIR_INCLUDE_DIR ${LLVM_MAIN_SRC_DIR}/../mlir/include)
  set(MLIR_GENERATED_INCLUDE_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)
  set(MLIR_INCLUDE_DIRS "${MLIR_INCLUDE_DIR};${MLIR_GENERATED_INCLUDE_DIR}")
endif()

set(TORCH_MLIR_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(TORCH_MLIR_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
message(STATUS "Building torch-mlir project at ${TORCH_MLIR_SOURCE_DIR} (into ${TORCH_MLIR_BINARY_DIR})")

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

function(torch_mlir_target_includes target)
  set(_dirs
    $<BUILD_INTERFACE:${MLIR_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${TORCH_MLIR_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${TORCH_MLIR_BINARY_DIR}/include>
  )
  # In LLVM parlance, the actual target may just be an interface and may not
  # be responsible for actually compiling anything. The corresponding obj.
  # target, when present, is just used for compilation and does not
  # contribute to the interface properties.
  # TODO: Normalize this upstream.
  target_include_directories(${target} PUBLIC ${_dirs})
  if(TARGET obj.${target})
    target_include_directories(obj.${target} PRIVATE ${_dirs})
  endif()
endfunction()

# Configure CMake.
list(APPEND CMAKE_MODULE_PATH ${MLIR_MAIN_SRC_DIR}/cmake/modules)
list(APPEND CMAKE_MODULE_PATH ${LLVM_MAIN_SRC_DIR}/cmake)

include(TableGen)
include(AddLLVM)
include(AddMLIR)

################################################################################
# Setup python.
################################################################################

if(MLIR_ENABLE_BINDINGS_PYTHON)
  include(MLIRDetectPythonEnv)
  mlir_configure_python_dev_packages()
endif()

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)

add_custom_target(check-torch-mlir-all)
add_dependencies(check-torch-mlir-all
  check-torch-mlir
  check-torch-mlir-dialects
  )

if(MLIR_ENABLE_BINDINGS_PYTHON)
  # If parent projects want to configure where to place the python packages,
  # respect that.
  if(NOT TORCH_MLIR_PYTHON_PACKAGES_DIR)
    set(TORCH_MLIR_PYTHON_PACKAGES_DIR "${CMAKE_CURRENT_BINARY_DIR}/python_packages")
  endif()
  add_dependencies(check-torch-mlir-all
  check-torch-mlir-python
  )
  add_subdirectory(python)
else()
  add_custom_target(TorchMLIRPythonModules)
endif()

add_subdirectory(test)
add_subdirectory(examples)
