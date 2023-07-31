add_llvm_executable(torch-mlir-capi-torch-test torch.c)
llvm_update_compile_flags(torch-mlir-capi-torch-test)
target_link_libraries(
        torch-mlir-capi-torch-test
        PRIVATE
        MLIRCAPIIR
        MLIRCAPIRegisterEverything
        TorchMLIRCAPI
)

add_lit_testsuite(check-torch-mlir-capi "Running the torch-mlir CAPI tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS torch-mlir-capi-torch-test FileCheck
        )
set_target_properties(check-torch-mlir-capi PROPERTIES FOLDER "Tests")
