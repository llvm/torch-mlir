add_llvm_executable(torch-mlir-capi-torch-test torch.c)
llvm_update_compile_flags(torch-mlir-capi-torch-test)
target_link_libraries(
        torch-mlir-capi-torch-test
        PRIVATE
        MLIRCAPIIR
        TorchMLIRCAPI
)
