message(STATUS "Enabling onnx_c_importer...")

include(FetchContent)

find_package(Protobuf REQUIRED CONFIG)

option(ONNX_DISABLE_EXCEPTIONS "For compatibility with LLVM build" ON)

FetchContent_Declare(
    onnx
    EXCLUDE_FROM_ALL
    GIT_REPOSITORY https://github.com/onnx/onnx.git
    GIT_TAG v1.15.0
    GIT_SHALLOW ON
    GIT_PROGRESS ON
)
FetchContent_MakeAvailable(onnx)

add_llvm_executable(
    torch-mlir-import-onnx
    PARTIAL_SOURCES_INTENDED

    import-onnx-main.cpp
    OnnxImporter.h
    OnnxImporter.cpp
)

target_link_libraries(
    torch-mlir-import-onnx
    LLVMSupport
    MLIRCAPIIR
    TorchMLIRCAPI
    onnx
)
