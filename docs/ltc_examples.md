# Torch-MLIR Lazy Tensor Core Backend Examples

Refer to the main documentation [here](ltc_backend.md).

## Example Usage
```python
import torch
import torch._lazy
import torch_mlir._mlir_libs._REFERENCE_LAZY_BACKEND as lazy_backend

# Register the example LTC backend.
lazy_backend._initialize()

device = 'lazy'

# Create some tensors and perform operations.
inputs = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.float32, device=device)
outputs = torch.tanh(inputs)

# Mark end of training/evaluation iteration and lower traced graph.
torch._lazy.mark_step()
print('Results:', outputs)

# Optionally dump MLIR graph generated from LTC trace.
computation = lazy_backend.get_latest_computation()
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

## Example Models
There are also examples of a [HuggingFace BERT](../examples/ltc_backend_bert.py) and [MNIST](../examples/ltc_backend_mnist.py) model running on the example LTC backend.
