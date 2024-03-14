import torch

from torch_mlir import fx
import sparse_test

def run(f):
    print(f"{f.__name__}")
    print("-" * len(f.__name__))
    f()
    print()

# @run
# def test_sigmoid():
#     class ElementwiseSigmoidModule(torch.nn.Module):
#         def __init__(self):
#             super().__init__()

#         def forward(self, x):
#             return torch.sigmoid(x)

#     m = sparse_test.sparse_jit(ElementwiseSigmoidModule(), torch.randn(128, 2))
    
#     print(m)

@run
def test_softmax():
    class SoftmaxModule(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.nn.functional.softmax(x, -1, torch.float32)

    output = sparse_test.sparse_jit(SoftmaxModule(), torch.randn(8, 128))

    print("-----------------------------------")
    print("torch dialect:")
    m = fx.export_and_import(SoftmaxModule(), torch.randn(8, 128))
    print(m)
    
