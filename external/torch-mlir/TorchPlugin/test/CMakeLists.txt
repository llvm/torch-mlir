configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(TEST_DEPENDS
  FileCheck count not
  torch-mlir-opt
  TorchMLIRPluginPythonModules
  )

add_lit_testsuite(check-torch-mlir-plugin
  "Running the torch-mlir-plugin regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${TEST_DEPENDS}
  )
set_target_properties(check-torch-mlir-plugin PROPERTIES FOLDER "Tests")

add_lit_testsuites(TORCH_MLIR_PLUGIN ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${TEST_DEPENDS})
