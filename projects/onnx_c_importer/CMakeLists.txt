message(STATUS "Enabling onnx_c_importer...")

include(FetchContent)

find_package(Protobuf REQUIRED CONFIG)

FetchContent_Declare(
    onnx
    EXCLUDE_FROM_ALL
    GIT_REPOSITORY https://github.com/onnx/onnx.git
    GIT_TAG v1.16.1
    GIT_SHALLOW ON
    GIT_PROGRESS ON
)
FetchContent_MakeAvailable(onnx)

set(LLVM_REQUIRES_EH ON)
set(LLVM_REQUIRES_RTTI ON)


add_llvm_executable(
    torch-mlir-import-onnx
    PARTIAL_SOURCES_INTENDED

    import-onnx-main.cpp
    OnnxImporter.h
    OnnxImporter.cpp
    SimpleArgParser.hpp
    Dict.hpp
    Status.hpp
    onnx_extras.hpp
)

set_target_properties(torch-mlir-import-onnx PROPERTIES CXX_STANDARD 20)

# Supress compiler warnings from onnx headers
check_cxx_compiler_flag(-Wno-c++98-compat-extra-semi
CXX_SUPPORTS_NO_CXX98_COMPAT_EXTRA_SEMI_FLAG)
if (CXX_SUPPORTS_CXX98_COMPAT_EXTRA_SEMI_FLAG)
    target_compile_options(torch-mlir-import-onnx PRIVATE
        "-Wno-c++98-compat-extra-semi")
    target_compile_options(onnx PRIVATE
        "-Wno-c++98-compat-extra-semi")
endif()

target_link_libraries(
    torch-mlir-import-onnx
    MLIRCAPIIR
    TorchMLIRCAPI
    onnx
)
