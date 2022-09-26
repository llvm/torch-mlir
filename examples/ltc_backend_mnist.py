# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.
"""
Example use of the example Torch MLIR LTC backend.
"""
import argparse
import sys

import numpy as np
import torch
import torch._lazy
import torch.nn.functional as F

from nest import visit_lazy_tensors


def main(device='lazy'):
    """
    Load model to specified device. Ensure that any backends have been initialized by this point.

    :param device: name of device to load tensors to
    """
    torch.manual_seed(0)

    inputs = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.float32, device=device)
    assert inputs.device.type == device

    targets = torch.tensor([3], dtype=torch.int64, device=device)
    assert targets.device.type == device

    assert lazy_backend.set_parameter_name(inputs, "input.0")
    assert lazy_backend.set_parameter_name(targets, "input.1")

    print("Initialized data")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(5, 10)

        def forward(self, x):
            out = self.fc1(x)
            out = F.relu(out)
            return out

    model = Model()

    parameters = [p.detach().numpy() for p in model.parameters()]

    model = model.to(device)
    model.train()
    assert all(p.device.type == device for p in model.parameters())

    print("Initialized model")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    num_epochs = 1
    losses = []

    print("Entering training loop")
    for _ in range(num_epochs):
        optimizer.zero_grad(set_to_none=True)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        losses.append(loss)

        print("Optimizer step")
        optimizer.step()

        if device == "lazy":
            print("Calling Mark Step", flush=True)
            torch._lazy.mark_step()

    # Get debug information from LTC
    if 'torch_mlir._mlir_libs._REFERENCE_LAZY_BACKEND' in sys.modules:
        computation = lazy_backend.get_latest_computation()
        if computation:
            print(computation.debug_string())

    print(losses)


    new_parameters = [p.detach().cpu().numpy() for p in model.parameters()]

    for p1, p2, in zip(parameters, new_parameters):
        assert not np.allclose(p1, p2)
        print(p1)
        print(p2)
        print()

    return model, losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--device",
        type=str.upper,
        choices=["CPU", "TS", "MLIR_EXAMPLE"],
        default="MLIR_EXAMPLE",
        help="The device type",
    )
    args = parser.parse_args()

    if args.device in ("TS", "MLIR_EXAMPLE"):
        if args.device == "TS":
            import torch._lazy.ts_backend
            torch._lazy.ts_backend.init()

        elif args.device == "MLIR_EXAMPLE":
            import torch_mlir._mlir_libs._REFERENCE_LAZY_BACKEND as lazy_backend

            lazy_backend._initialize()

        device = "lazy"
        print("Initialized backend")
    else:
        device = args.device.lower()

    main(device)
