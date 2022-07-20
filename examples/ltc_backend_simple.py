import torch
import torch._lazy
import torch_mlir.reference_lazy_backend._REFERENCE_LAZY_BACKEND as lazy_backend

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
