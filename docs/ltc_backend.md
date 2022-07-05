# Torch-MLIR Lazy Tensor Core Backend

## Introduction
[Lazy Tensor Core](https://github.com/pytorch/pytorch/blob/master/torch/csrc/lazy/tutorial.md) is a tracing system in PyTorch which is supported as an entry point to Torch-MLIR.
After registering an LTC backend, all operations performed on lazy tensors are recorded and handed off to the backend implementation.

Lazy Tensor Core support is provided through an abstract [`TorchMlirBackendImpl`](../python/torch_mlir/csrc/base_lazy_backend/backend_impl.h) class, which handles the conversion to MLIR.
Implementations based on this abstract class will be able to configure their own compile and execution workflows.
An example implementation is available [here](../examples/ltc_backend/ltc_backend), and additional details about how to implement a custom backend is available [below](#Implementing-a-custom-backend).

### Example Usage
```python
import torch
import torch._lazy
import ltc_backend.ltc_backend._EXAMPLE_MLIR_BACKEND as ltc_backend

# Register the example LTC backend.
ltc_backend._initialize()

device = 'lazy'

# Create some tensors and perform operations.
inputs = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.float32, device=device)
outputs = torch.tanh(inputs)

# Mark end of training iteration and trace graph.
torch._lazy.mark_step()
print('Results: ', outputs)

# Dump MLIR graph generated from LTC trace. 
computation = ltc_backend.get_latest_computation()
if computation:
    print(computation.debug_string())
```

```
Received 1 computation instances at Compile!
Received 1 arguments, and returned 2 results during ExecuteCompile!

Results: tensor([[0.7616, 0.9640, 0.9951, 0.9993, 0.9999]], device='lazy:0')

JIT Graph: 
graph(%p0 : Float(1, 5)):
  %1 : Float(1, 5) = aten::tanh(%p0)
  return (%p0, %1)

MLIR: 
func.func @graph(%arg0: !torch.vtensor<[1,5],f32>) -> (!torch.vtensor<[1,5],f32>, !torch.vtensor<[1,5],f32>) {
  %0 = torch.aten.tanh %arg0 : !torch.vtensor<[1,5],f32> -> !torch.vtensor<[1,5],f32>
  return %arg0, %0 : !torch.vtensor<[1,5],f32>, !torch.vtensor<[1,5],f32>
}

Input/Output Alias Mapping: 
Output: 0 -> Input param: 0

In Mark Step: true
```

### Example Models
There are also examples of a [HuggingFace BERT](../examples/ltc_backend_bert.py) and [MNIST model](../examples/ltc_backend_mnist.py) running on the example/reference LTC backend.

## Code Structure

## Implementing a custom backend
