function(ProbeForPyTorchInstall)
  if(Torch_ROOT)
    message(STATUS "Using cached Torch root = ${Torch_ROOT}")
  else()
    message(STATUS "Checking for PyTorch using ${PYTHON_EXECUTABLE} ...")
    execute_process(
      COMMAND ${PYTHON_EXECUTABLE}
      -c "import os;import torch;print(os.path.dirname(torch.__file__), end='')"
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      RESULT_VARIABLE PYTORCH_STATUS
      OUTPUT_VARIABLE PYTORCH_PACKAGE_DIR)
    if(NOT PYTORCH_STATUS EQUAL "0")
      message(FATAL_ERROR "Unable to 'import torch' with ${PYTHON_EXECUTABLE}")
    endif()
    message(STATUS "Found PyTorch installation at ${PYTORCH_PACKAGE_DIR}")

    # PyTorch stashes its installed .cmake files under share/cmake/Torch.
    set(Torch_ROOT "${PYTORCH_PACKAGE_DIR}/share/cmake/Torch"
        CACHE STRING "Torch package root")
  endif()
endfunction()
