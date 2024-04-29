# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.
"""
Example use of the example Torch MLIR LTC backend.
"""
import argparse
import sys

import torch
import torch._lazy
import torch.nn.functional as F


def main(device="lazy"):
    """
    Load model to specified device. Ensure that any backends have been initialized by this point.

    :param device: name of device to load tensors to
    """
    torch.manual_seed(0)

    inputs = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.float32, device=device)
    assert inputs.device.type == device

    targets = torch.tensor([3], dtype=torch.int64, device=device)
    assert targets.device.type == device

    print("Initialized data")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(5, 10)

        def forward(self, x):
            out = self.fc1(x)
            out = F.relu(out)
            return out

    model = Model().to(device)
    model.train()
    assert all(p.device.type == device for p in model.parameters())

    print("Initialized model")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    num_epochs = 3
    losses = []
    for _ in range(num_epochs):
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        losses.append(loss)

        optimizer.step()

        if device == "lazy":
            print("Calling Mark Step")
            torch._lazy.mark_step()

    # Get debug information from LTC
    if "torch_mlir._mlir_libs._REFERENCE_LAZY_BACKEND" in sys.modules:
        computation = lazy_backend.get_latest_computation()
        if computation:
            print(computation.debug_string())

    print(losses)

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
