add_subdirectory(builder)

include(NpcompPython)

# Sharp edge: Torch extensions need to use the same pybind11 that torch
# was compiled with, or else there will be issues in cross module exception
# handling (which will abort instead of raise). We circumvent the possibility
# by forcing the torch directories first.
include_directories(BEFORE
  ${TORCH_INCLUDE_DIRS}
  ${TORCH_INSTALL_PREFIX}/include/TH
  ${TORCH_INSTALL_PREFIX}/include/THC/opt/pytorch/pytorch
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${Python3_INCLUDE_DIRS}
  # TODO: Fix implicit ordering. If PyTorch was build against an external
  # pybind11, then it will not be in the above search path and must be
  # resolved here, in the hope that it is the same one we were configured
  # with (which it should be if installed via pip). This is really fragile,
  # though, causing cast failures at runtime if we get it wrong. Come up with
  # a way to strengthen this.
  ${pybind11_INCLUDE_DIR}
  )
link_directories("${TORCH_INSTALL_PREFIX}/lib")

add_library(NPCOMPTorchMLIRExt SHARED
  init_python_bindings.cpp
  )
target_link_libraries(NPCOMPTorchMLIRExt
  ${TORCH_LIBRARIES}
  ${Python3_LIBRARIES}
  torch_python
  npcomp_torch_builder_bindings

  # NPCOMP shared library.
  NPCOMP
)
add_dependencies(NPCOMPTorchMLIRExt
  # Uses of the torch_mlir extension also require the npcomp extension to
  # be built.
  NPCOMPNativePyExt
)
set_target_properties(NPCOMPTorchMLIRExt PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/python
  OUTPUT_NAME _torch_mlir
  PREFIX "${PYTHON_MODULE_PREFIX}"
  SUFFIX "${PYTHON_MODULE_EXTENSION}"
  CXX_VISIBILITY_PRESET "hidden"
  )

npcomp_python_target_compile_options(NPCOMPTorchMLIRExt)
mlir_check_all_link_libraries(NPCOMPTorchMLIRExt)
