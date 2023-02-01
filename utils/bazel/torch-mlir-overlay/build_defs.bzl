"""custom extension for bazel tools"""

load("@bazel_skylib//lib:paths.bzl", "paths")

# These flags are needed for pybind11 to work.
PYBIND11_COPTS = [
    "-fexceptions",
    "-frtti",
]

PYBIND11_FEATURES = [
    # Cannot use header_modules (parse_headers feature fails).
    "-use_header_modules",
]

def _gen_python_package_impl(ctx):
    out = []
    for src in ctx.files.py_srcs:
        mlir_prefix = "external/llvm-project/mlir/python/mlir/"
        torch_mlir_prefix = "external/torch-mlir/python/torch_mlir/"
        torch_mlir_end2end_prefix = "external/torch-mlir/python/torch_mlir_e2e_test/"
        dst_suffix = ""
        if src.path.find(mlir_prefix) != -1:
            dst_suffix = src.path[src.path.find(mlir_prefix) + len(mlir_prefix):]
            dst_path = ctx.attr.name + "/torch_mlir/torch_mlir/" + dst_suffix
        elif src.path.find(torch_mlir_prefix) != -1:
            dst_suffix = src.path[src.path.find(torch_mlir_prefix) + len(torch_mlir_prefix):]
            dst_path = ctx.attr.name + "/torch_mlir/torch_mlir/" + dst_suffix
        elif src.path.find(torch_mlir_end2end_prefix) != -1:
            dst_suffix = src.path[src.path.find(torch_mlir_end2end_prefix) + len(torch_mlir_end2end_prefix):]
            dst_path = ctx.attr.name + "/torch_mlir/torch_mlir_e2e_test/" + dst_suffix
        else:
            print("%s has no such prefix %s and %s" % (src.path, mlir_prefix, torch_mlir_prefix, torch_mlir_end2end_prefix))
            continue
        dst_file = ctx.actions.declare_file(dst_path)
        ctx.actions.run_shell(
            inputs = [src],
            outputs = [dst_file],
            arguments = [
                src.path,
                dst_file.path,
            ],
            progress_message = "Copying py file %s into '%s'" % (src.path, dst_file.path),
            command = "cp $1 $2",
        )
        out.append(dst_file)

    src = ctx.attr.so_srcs
    lc = src[CcInfo].linking_context
    count = 0
    for linker_input in lc.linker_inputs.to_list():
        for library in linker_input.libraries:
            if count == 0:
                dst_path = ctx.attr.name + "/torch_mlir/torch_mlir/_mlir_libs/" + library.dynamic_library.basename
            else:  # for deps
                lib_dir = library.dynamic_library.dirname.split("/")[-1] + "/"
                if lib_dir == "_solib_k8/":
                    lib_dir = ""
                dst_path = ctx.attr.name + "/torch_mlir/torch_mlir/__main__/_solib_k8/" + lib_dir + library.dynamic_library.basename
            dst_file = ctx.actions.declare_file(dst_path)
            ctx.actions.run_shell(
                inputs = [library.dynamic_library],
                outputs = [dst_file],
                arguments = [
                    library.dynamic_library.path,
                    dst_file.path,
                ],
                progress_message = "Copying so file %s into '%s'" % (library.dynamic_library.path, dst_path),
                command = "cp $1 $2",
            )
            out.append(dst_file)
        count += 1

    return [DefaultInfo(files = depset(out))]

gen_python_package = rule(
    implementation = _gen_python_package_impl,
    attrs = {
        "py_srcs": attr.label_list(mandatory = True, allow_files = True),
        "so_srcs": attr.label(mandatory = True, allow_files = True),
    },
)

def target_path(label):
    """Returns the path to the 'label'.

    Args:
      label: label. The label to return the target path of.

    For example, target_path("@foo//bar:baz") returns 'external/foo/bar/baz'.
    """

    return paths.join(Label(label).workspace_root, Label(label).package, Label(label).name)
