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

project(torch_mlir_plugin LANGUAGES CXX C)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 14)

#-------------------------------------------------------------------------------
# Setup PyTorch
#-------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")
include(TorchMLIRPyTorch)
TorchMLIRProbeForPyTorchInstall()
find_package(Torch 1.8 REQUIRED)

TorchMLIRConfigurePyTorch()

#-------------------------------------------------------------------------------
# Subdirectories
#-------------------------------------------------------------------------------

add_subdirectory(csrc)
add_subdirectory(python)
add_subdirectory(test)
