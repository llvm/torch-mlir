include_directories(
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
add_library(npcomp_torch_builder_bindings
  acap_dispatch.cpp
  debug.cpp
  func_builder.cpp
  graph_importer.cpp
  module_builder.cpp
  python_bindings.cpp
)

target_link_libraries(npcomp_torch_builder_bindings
  ${TORCH_LIBRARIES}
  ${Python3_LIBRARIES}
  torch_python
  )
