llvm_canonicalize_cmake_booleans(
  MLIR_ENABLE_BINDINGS_PYTHON
  TORCH_MLIR_ENABLE_REFBACKEND
  TORCH_MLIR_ENABLE_STABLEHLO
)

configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(TORCH_MLIR_TEST_DEPENDS
        FileCheck count not
        TorchMLIRPythonModules
        torch-mlir-opt
        torch-mlir-capi-torch-test
        )

add_lit_testsuite(check-torch-mlir "Running the torch-mlir regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${TORCH_MLIR_TEST_DEPENDS}
        )
set_target_properties(check-torch-mlir PROPERTIES FOLDER "Tests")

add_lit_testsuites(TORCH_MLIR ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${TORCH_MLIR_TEST_DEPENDS})

add_subdirectory(CAPI)
