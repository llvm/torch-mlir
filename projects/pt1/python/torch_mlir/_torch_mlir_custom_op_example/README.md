# Torch-mlir custom op example

This library is a stand-in for a PyTorch C++ extension (see [this PyTorch Tutorial](https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html) for more information).
If you're reading this, you're likely looking to create or support a third-party PyTorch extension and add torch-mlir support to it.
This isn't much different than [adding a new PyTorch op to torch-mlir](https://github.com/llvm/torch-mlir/wiki/Torch-ops-E2E-implementation).
You'll still go through the exact same process, with just a small change:

 - Before running `update_torch_ods.sh` or `update_abstract_interp_lib.sh`, you'll want to set the `TORCH_MLIR_EXT_PYTHONPATH` to point to wherever your extension lives and the `TORCH_MLIR_EXT_MODULES` to the name of the python module.

For instance, let's say you've written a python package called `my_torch_ops`.
If `my_torch_ops` lives in `/example/subdirectory/`, then you'll want to set `TORCH_MLIR_EXT_PYTHONPATH=/example/subdirectory` and `TORCH_MLIR_EXT_MODULES=my_torch_ops`.
If you've installed your package (with `pip`, for instance), then you'll only need to set `TORCH_MLIR_EXT_MODULES=my_torch_ops`.
Note that the `update_torch_ods.sh` and `update_abstract_interp_lib.sh` scripts do not use the `PYTHONPATH` environment variable in your current shell.
This is on purpose, but it means that you either need to set `TORCH_MLIR_EXT_PYTHONPATH` to include your package or to include the paths set in your shell's `PYTHONPATH` variable.
If you have more than one PyTorch extension, you can add them all by including each path in `TORCH_MLIR_EXT_PYTHONPATH` separated by colons (`:`) and each module in `TORCH_MLIR_EXT_MODULES` separated by commas (`,`).

**It is important that your custom ops are registered with PyTorch as a side-effect of importing your extension.**
(The `__init__.py` file in this directory is an example of how to do this.)

Also, while this example library is a component of torch-mlir, in general this is not necessary or desirable.
As long as your new ops are registered with PyTorch as a side effect of loading your Python module, then it can reside anywhere and doesn't really even need to be written to know about torch-mlir at all.
Torch-mlir itself must be statically modified to support your ops using the normal process, but that support won't negatively affect torch-mlir users who do not use your extension.
