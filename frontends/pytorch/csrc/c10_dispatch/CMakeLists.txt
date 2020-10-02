include_directories(
  ${TORCH_INCLUDE_DIRS}
  ${TORCH_INSTALL_PREFIX}/include/TH
  ${TORCH_INSTALL_PREFIX}/include/THC/opt/pytorch/pytorch
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${PYTHON_INCLUDE_DIRS}
  )
link_directories("${TORCH_INSTALL_PREFIX}/lib")
add_library(npcomp_torch_c10_dispatch_bindings
  acap_dispatch.cpp
  func_builder.cpp
  module_builder.cpp
  python_bindings.cpp
)

get_property(mlir_libs GLOBAL PROPERTY MLIR_ALL_LIBS)
target_link_libraries(npcomp_torch_c10_dispatch_bindings
  NPCOMPCAPIIR
  ${TORCH_LIBRARIES}
  ${PYTHON_LIBRARIES}
  ${mlir_libs}
  torch_python
  )
