# TorchMLIRProbeForPyTorchInstall
# Attempts to find a Torch installation and set the Torch_ROOT variable
# based on introspecting the python environment. This allows a subsequent
# call to find_package(Torch) to work.
function(TorchMLIRProbeForPyTorchInstall)
  if(Torch_ROOT)
    message(STATUS "Using cached Torch root = ${Torch_ROOT}")
  else()
    message(STATUS "Checking for PyTorch using ${Python3_EXECUTABLE} ...")
    execute_process(
      COMMAND ${Python3_EXECUTABLE}
      -c "import os;import torch;print(torch.utils.cmake_prefix_path, end='')"
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      RESULT_VARIABLE PYTORCH_STATUS
      OUTPUT_VARIABLE PYTORCH_PACKAGE_DIR)
    if(NOT PYTORCH_STATUS EQUAL "0")
      message(STATUS "Unable to 'import torch' with ${Python3_EXECUTABLE} (fallback to explicit config)")
      return()
    endif()
    message(STATUS "Found PyTorch installation at ${PYTORCH_PACKAGE_DIR}")

    set(Torch_ROOT "${PYTORCH_PACKAGE_DIR}" CACHE STRING
        "Torch configure directory" FORCE)
  endif()
endfunction()

# TorchMLIRConfigurePyTorch
# Performs configuration of PyTorch flags after CMake has found it to be
# present. Most of this comes down to detecting whether building against a
# source or official binary and adjusting compiler options in the latter case
# (in the former, we assume that it was built with system defaults). We do this
# conservatively and assume non-binary builds by default.
#
# In the future, we may want to switch away from custom building these
# extensions and instead rely on the Torch machinery directly (definitely want
# to do that for official builds).
function(TorchMLIRConfigurePyTorch)
  if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    # Linux specific libstdcpp ABI checking.
    message(STATUS "Checking if Torch is an official binary ...")
    execute_process(
      COMMAND ${Python3_EXECUTABLE}
      -c "from torch.utils import cpp_extension as c; import sys; sys.exit(0 if c._is_binary_build() else 1)"
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      RESULT_VARIABLE _is_binary_build)
    if(${_is_binary_build} EQUAL 0)
      set(TORCH_CXXFLAGS "")
      if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
        set(TORCH_CXXFLAGS "-D_GLIBCXX_USE_CXX11_ABI=0 -fabi-version=11")
      elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
        set(TORCH_CXXFLAGS "-D_GLIBCXX_USE_CXX11_ABI=0 -U__GXX_ABI_VERSION -D__GXX_ABI_VERSION=1011 '-DPYBIND11_COMPILER_TYPE=\"_gcc\"'")
      else()
        message(WARNING "Unrecognized compiler. Cannot determine ABI flags.")
        return()
      endif()
      message(STATUS "Detected Torch official binary build. Setting ABI flags: ${TORCH_CXXFLAGS}")
      set(TORCH_CXXFLAGS "${TORCH_CXXFLAGS}" PARENT_SCOPE)
    endif()
  endif()
endfunction()

function(torch_mlir_python_target_compile_options target)
  target_compile_options(${target} PRIVATE
    $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
      # Enable RTTI and exceptions.
      -frtti -fexceptions
      # Noisy pybind warnings
      -Wno-unused-value
      -Wno-covered-switch-default
    >
    $<$<CXX_COMPILER_ID:MSVC>:
    # Enable RTTI and exceptions.
    /EHsc /GR>
  )
endfunction()
