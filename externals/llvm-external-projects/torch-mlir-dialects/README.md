# Torch-MLIR Dialects Project

Sources for torch-mlir's public dialects (containing ops/types/attributes that
are unique to Torch-MLIR at the moment)

This project is intended to be used via LLVM's external projects setup:

* `-DLLVM_EXTERNAL_PROJECTS=torch-mlir-dialects`
* `-DLLVM_EXTERNAL_TORCH_MLIR_DIALECTS_SOURCE_DIR={this_directory}`

It depends on the `mlir` project.

## TCP Dialect
Tensor Compute Primitives (TCP) is a mid-level transformation oriented IR for deep learning & similar applications.

TCP is being bootstrapped under Torch-MLIR Dialects.

All technical discussions regarding TCP will be done in the discourse [TCP-WG](https://discourse.llvm.org/c/mlir/mlir-tcp-wg/36) category.
### References
* Discourse [thread](https://discourse.llvm.org/t/rfc-proposal-for-a-high-level-ml-dialect-in-mlir/64249) regarding TCP proposal.
* TCP [proposal document](https://docs.google.com/document/d/1f3KVsXA4xm6W7gd2cKx9ThGA52XMfZKsPV3ZOlsztC4/edit)
* TCP spec [draft](https://docs.google.com/document/d/1Twyph8jU_f1QDoBInr8OkUXxcyRw-KqFdyclkK1UQbc/edit)
* Torch-MLIR [issue](https://github.com/llvm/torch-mlir/issues/1366) to track bootstrapping TCP.
