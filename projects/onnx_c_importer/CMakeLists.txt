message(STATUS "Enabling onnx_c_importer...")

include(FetchContent)

find_package(Protobuf)
if(NOT Protobuf_FOUND)
    message(FATAL_ERROR 
        "In order to build C ONNX support, the Protobuf package must be installed "
        "on the system. Without this ONNX will attempt to build it in the project "
        "and the dependent ABSEIL build system is incompatible. "
        "On Ubuntu, install with: "
        "apt install libprotobuf-dev protobuf-compiler\n\n"
        "(or this entire component can be disabled with "
        "-DTORCH_MLIR_ENABLE_ONNX_C_IMPORTER=OFF)")
endif()

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
