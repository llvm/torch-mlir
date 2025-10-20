# Adding an E2E Test

## Overview

Adding support for a Torch operator in Torch-MLIR should always be accompanied
by at least one end-to-end test to make sure the implementation of the op
matches the behavior of PyTorch. The tests live in the
`torch-mlir/projects/e2e/torch_mlir_e2e_test/test_suite` directory. When adding a new
test, choose a file that best matches the op you're testing, and if there is no
file that best matches add a new file for your op.

## An E2E Test Deconstructed

In order to understand how to create an end-to-end test for your op, let's break
down an existing test to see what the different parts mean:

```python
class IndexTensorModule3dInput(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
        ([-1, -1], torch.int64, True),
    ])
    def forward(self, x, index):
        return torch.ops.aten.index(x, (index,))


@register_test_case(module_factory=lambda: IndexTensorModule3dInput())
def IndexTensorModule3dInput_basic(module, tu: TestUtils):
    module.forward(tu.rand(5, 4, 3), tu.randint(2, 3, high=3))
```

### Class Name


```python
class IndexTensorModule3dInput(torch.nn.Module):
```

The class name should always contain the name of the op that is being
tested. This makes it easy to search for tests for a particular op. Often times
an op will require multiple tests to make sure different paths in the
compilation work as expected. In such cases, it is customary to add extra
information to the class name about what is being tested. In this example, the
op is being tested with a rank-3 tensor as an input.

### `__init__` Method

```python
    def __init__(self):
        super().__init__()
```

In most tests, the `__init__` method simply calls the `__init__` method of the
`torch.nn.Module` class. However, sometimes this method can be used to
initialize parameters needed in the `forward` method. An example of such a case
is in the [E2E test for Resnet18](https://github.com/llvm/torch-mlir/blob/ba17a4d6c09b4bbb4ef21b1d8d4a93cb056be109/python/torch_mlir_e2e_test/test_suite/vision_models.py#L17-L22).


### `@export` and `@annotate_args` Decorators

```python
    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
        ([-1, -1], torch.int64, True),
    ])
```

The [`@export` decorator](https://github.com/llvm/torch-mlir/blob/ba17a4d6c09b4bbb4ef21b1d8d4a93cb056be109/python/torch_mlir_e2e_test/torchscript/annotations.py#L30)
lets the importer know which methods in the class will be public after the
`torch.nn.Module` gets imported into the `torch` dialect. All E2E tests should
have this decorator on the `forward` method.

The [`@annotate_args` decorator](https://github.com/llvm/torch-mlir/blob/ba17a4d6c09b4bbb4ef21b1d8d4a93cb056be109/python/torch_mlir_e2e_test/torchscript/annotations.py#L53)
is used to give the importer information about the arguments of the method being
decorated, which can then be propagated further into the IR of the body of the
method. The list of annotations **must** have one annotation for each argument
including the `self` argument. The `self` argument always gets the annotation of
`None`, while the other inputs get an annotation with three fields in the
following order:

1. Shape of input tensor. Use `-1` for dynamic dimensions
2. Dtype of the input tensor
3. Boolean representing whether the input tensor [has value semantics](https://github.com/llvm/torch-mlir/blob/main/projects/jit_ir_common/csrc/jit_ir_importer/class_annotator.h#L54-L67). This
   will always be true for E2E tests, since the [Torch-MLIR backend contract](architecture.md#the-backend-contract) requires all tensors in the
   IR to eventually have value semantics.

From the structure of the annotations for the arguments other than the `self`
argument it is clear that only tensor arguments are supported. This means that
if an op requires an input other than a tensor, you need to do one of the
following:

- Create the value in the method body
- Create the value as a class parameter in the `__init__` method
- In the case of certain values such as `int`s and `float`s, you can pass a
  zero-rank tensor as an input and use `int(input)` or `float(input)`in the
  method body to turn the tensor into a scalar `int` or `float`, respectively.

### `forward` Method

```python
    def forward(self, x, index):
        return torch.ops.aten.index(x, (index,))
```

The forward method should be a simple test of your op. In other words, it will
almost always take the form of simply returning the result of calling your
op. The call to your op should **always** be made using
`torch.ops.aten.{op_name}` to make it very clear which ATen op is being
tested. Some ATen ops have different variants under the same base name, such as
`aten.mean`, which has also a variant `aten.mean.dim`. At the Python level, such
ops are accessed by just their base name, and the right variant is chosen based
on the inputs given. For example, to test `aten.mean.dim` the test should use
`torch.ops.aten.mean(..., dim=...)`.

### `@register_test_case` Decorator

```python
@register_test_case(module_factory=lambda: IndexTensorModule3dInput())
```

The `@register_test_case` decorator is used to register the test case
function. The `module_factory` argument should be a function that when called
produces an instance of the test class. This function will be used to create the
first argument passed to the test case function.

### Test Case Function

```python
def IndexTensorModule3dInput_basic(module, tu: TestUtils):
    module.forward(tu.rand(5, 4, 3), tu.randint(2, 3, high=3))
```

The convention adopted for the name of the test case function is to have the
same name as the test class postfixed by `_basic`. The test function always
takes an instance of the test class as the first argument and a
[`TestUtils`](https://github.com/llvm/torch-mlir/blob/8e880a2d009b67d45fb07434ab62ec2066a11185/python/torch_mlir_e2e_test/torchscript/framework.py#L167)
object as the second argument. The `TestUtils` has some methods, such as
[`tu.rand`](https://github.com/llvm/torch-mlir/blob/8e880a2d009b67d45fb07434ab62ec2066a11185/python/torch_mlir_e2e_test/torchscript/framework.py#L182)
and
[`tu.randint`](https://github.com/llvm/torch-mlir/blob/8e880a2d009b67d45fb07434ab62ec2066a11185/python/torch_mlir_e2e_test/torchscript/framework.py#L185),
that allow the creation of random tensors in a way that makes sure the compiled
module and the golden trace receive the same tensors as input. Therefore, all
random inputs should be generated through the `TestUtils` object.


## Things to Consider When Creating New Tests

- Do you need negative numbers? If so,
  [`tu.rand`](https://github.com/llvm/torch-mlir/blob/8e880a2d009b67d45fb07434ab62ec2066a11185/python/torch_mlir_e2e_test/torchscript/framework.py#L182)
  and
  [`tu.randint`](https://github.com/llvm/torch-mlir/blob/8e880a2d009b67d45fb07434ab62ec2066a11185/python/torch_mlir_e2e_test/torchscript/framework.py#L185)
  both allow you to specify a lower and upper bound for random number generation
- Make sure the annotation of the forward method matches the input types and
  shapes
- If an op takes optional flag arguments, there should be a test for each flag
  that is supported
- If there are tricky edge cases that your op needs to handle, have a test for
  each edge case
- Always follow the style and conventions of the file you're adding a test
  in. An attempt has been made to keep all E2E test files with consistent style,
  but file specific variations do exist

## Special kinds of tests

The testing of functions that produce random values (e.g. `torch.rand`) is
supported by our e2e test suite. The basic approach is that you generate a
"sufficiently large" random sample and then take a statistic (such as mean or
standard deviation) and compare it to the analytically expected value. For a
sufficiently large random sample, the test will be non-flaky. However, try to
avoid excessively large random samples, since our end-to-end test suite
currently runs on the RefBackend and so it can be very slow and
memory-inefficient to operate on large data. See examples in
[test_suite/rng.py](https://github.com/llvm/torch-mlir/blob/6c5360e281f31059f9c565e9ccc0f6edaa2c9a69/python/torch_mlir_e2e_test/test_suite/rng.py#L1).

The testing of functions with special numerical precision considerations can
also be tricky. Our rule of thumb is that if a test would fail across two
upstream PyTorch backends (e.g. CPU and CUDA) due to different numerical
precision choices, then it should not be included in our e2e test suite.
See [this PR](https://github.com/llvm/torch-mlir/pull/1605) for context.
