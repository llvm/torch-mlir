package(default_visibility = ["//visibility:public"])\

cc_library(
    name = "torch",
    srcs = select({
        "@platforms//os:windows": [
            "torch/lib/torch.lib",
            "torch/lib/torch_cpu.lib",
            "torch/lib/torch_global_deps.dll",
            "torch/lib/c10.lib",
        ],
        "//conditions:default": glob([
            "torch/lib/libshm.so",
            "torch/lib/libtorch.so",
            "torch/lib/libtorch_cpu.so",
            "torch/lib/libtorch_python.so",
            "torch/lib/libtorch_global_deps.so",
            "torch/lib/libc10.so",
            "torch/lib/libgomp*",
        ]),
    }),
    hdrs = glob([
        "torch/include/**/*.h",
    ]),
    includes = [
        "torch/include",
        "torch/include/torch/csrc/api/include/",
    ],
    deps = [
        "@local_config_python//:python_headers",
    ],
)
