# lit needs to find the tools, and the location differs for in-tree and out-of-tree builds.
get_target_property(TORCH_MLIR_DIALECTS_TOOLS_DIR torch-mlir-dialects-opt RUNTIME_OUTPUT_DIRECTORY)
configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(TORCH_MLIR_DIALECTS_TEST_DEPENDS
        FileCheck count not
        torch-mlir-dialects-opt
        )

add_lit_testsuite(check-torch-mlir-dialects "Running the torch-mlir-dialects regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${TORCH_MLIR_DIALECTS_TEST_DEPENDS}
        )
set_target_properties(check-torch-mlir-dialects PROPERTIES FOLDER "Tests")

add_lit_testsuites(TORCH_MLIR_DIALECTS ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${TORCH_MLIR_DIALECTS_TEST_DEPENDS})
