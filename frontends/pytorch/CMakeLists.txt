#-------------------------------------------------------------------------------
# Sub project setup
#-------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.13.4)

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

project(npcomp_pytorch LANGUAGES CXX C)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 14)

#-------------------------------------------------------------------------------
# Output paths
#-------------------------------------------------------------------------------

if(NOT MLIR_NPCOMP_PYTHON_PACKAGES_DIR)
  set(MLIR_NPCOMP_PYTHON_PACKAGES_DIR "${CMAKE_CURRENT_BINARY_DIR}/python_packages")
endif()

#-------------------------------------------------------------------------------
# Subdirectories
#-------------------------------------------------------------------------------

add_subdirectory(python)
add_subdirectory(test)
