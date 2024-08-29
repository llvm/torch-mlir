import os
import torch

# Register _torch_mlir_custom_op_example.identity as a side-effect of importing.
current_dir = os.path.dirname(os.path.abspath(__file__))
lib = os.path.join(*[current_dir, "libtorch_mlir_custom_op_example.so"])
torch.ops.load_library(lib)
