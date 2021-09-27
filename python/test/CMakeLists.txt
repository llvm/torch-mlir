configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(TEST_DEPENDS
  FileCheck count not
  torch-mlir-opt
  TorchMLIRPythonModules
  )

add_lit_testsuite(check-torch-mlir-python "Running the torch-mlir Python regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${TEST_DEPENDS}
  )
set_target_properties(check-torch-mlir-python PROPERTIES FOLDER "Tests")

add_lit_testsuites(TORCH_MLIR_PYTHON ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${TEST_DEPENDS})
add_dependencies(check-torch-mlir-all check-torch-mlir-python)
