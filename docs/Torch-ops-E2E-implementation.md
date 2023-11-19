## Tutorials
- Linalg.generic op introduction [here](https://www.youtube.com/watch?v=PdQPlPudT90)
- Basic E2E debugging walk-through [here](https://www.youtube.com/watch?v=E5epCJOtrf8&t=1556s)

## Example PR
https://github.com/llvm/torch-mlir/pull/294

## Major steps

### Step 1. Add an end-to-end test to iterate on

Add an end-to-end test to the [end-to-end test suite](https://github.com/llvm/torch-mlir/blob/main/docs/adding_an_e2e_test.md)). Ideally there is an existing file that your op fits into. If not, you can create a new file.

We generally recommend testing by invoking `torch.ops.aten.someop` from Python -- that gives a very precise test for the individual Torch operator you are implementing (calling `torch.ops.aten.someop` from Python always lowers into the MLIR `torch.aten.someop` operation)

The end-to-end test is important to check the correctness of the other steps.

### Step 2. Update ods

Update [torch_ods_gen.py](https://github.com/llvm/torch-mlir/blob/main/projects/pt1/python/torch_mlir/jit_ir_importer/build_tools/torch_ods_gen.py) with the new op and run [update_torch_ods.sh](https://github.com/llvm/torch-mlir/blob/main/build_tools/update_torch_ods.sh) to generate the ods. Running `update_torch_ods.sh` would dump all the operators with schema into `JITOperatorRegistryDump.txt`. It’s convenient to look for ops signatures and operands names in this file.

### Step 3. Propagate types
It’s essential to make sure the new op implements shape and dtype inference. See [abstract_interp_lib](https://github.com/llvm/torch-mlir/blob/main/docs/abstract_interp_lib.md) for information on adding shape and dtype inference.

### Step 4. Torch ops lowering

#### Decompose


If your op can be decomposed into other supported ops, then you can add a pattern into [DecomposeComplexOps](https://github.com/llvm/torch-mlir/blob/8d3ca887df5ac5126fa3fc2ec3546c6322a4d066/lib/Dialect/Torch/Transforms/DecomposeComplexOps.cpp#L1).

You can find example PRs [here](https://github.com/llvm/torch-mlir/pull/2550) and [here](https://github.com/llvm/torch-mlir/pull/2553).

#### Lower to Linalg

The `Torch` dialect needs to be lowered to [Linalg](https://mlir.llvm.org/docs/Dialects/Linalg/) dialect which can be used as input IR of backends. [Here](https://mlir.llvm.org/docs/Dialects/Linalg/#high-level-description-of-linalg-opsa-namelinalg_opsa) is a high level introduction about Linalg ops and [here](https://www.youtube.com/watch?v=PdQPlPudT90) is a video explaining `linalg.generic` op. The building block is the `linalg.generic` op which consists of indexing maps, iterator types, input/output tensors and a compute payload. You would want to get familiar with the concept of [affine map](https://mlir.llvm.org/docs/Dialects/Affine/#affine-expressions). The `linalg.generic` op anatomy [tutorial](https://www.youtube.com/watch?v=PdQPlPudT90&list=PLHPjgRtRcfTpVGFMrLP2KQyXhvtSQiiai&index=1) covers the basics of `linalg.generic` from a user's perspective.

You can find an [example PR here](https://github.com/llvm/torch-mlir/pull/294).

## Delivering Code
1. The codebase follows the [LLVM’s coding conventions](https://llvm.org/docs/CodingStandards.html).The following items might be the most frequently used rules:
- [use-early-exits-and-continue-to-simplify-code](https://llvm.org/docs/CodingStandards.html#use-early-exits-and-continue-to-simplify-code)
- [don-t-use-else-after-a-return](https://llvm.org/docs/CodingStandards.html#don-t-use-else-after-a-return)
- [use-auto-type-deduction-to-make-code-more-readable](https://llvm.org/docs/CodingStandards.html#use-auto-type-deduction-to-make-code-more-readable)
- [anonymous-namespaces](https://llvm.org/docs/CodingStandards.html#anonymous-namespaces)
- [avoid-braces-on-simple-single-statement-bodies-of-if-else-loop-statements](https://llvm.org/docs/CodingStandards.html#don-t-use-braces-on-simple-single-statement-bodies-of-if-else-loop-statements)
2. Try to refactor and reuse existing code/helpers when working on RefineTypes and TorchToLinalg lowering for easier maintenance, testing and better readability. Try not to copy & paste existing code.
3. Squash all the commits into one, including the commits addressing review comments.
4. Use `git clang-format HEAD~1` to automatically format your commit.
5. Rebase on `HEAD` before delivering.
