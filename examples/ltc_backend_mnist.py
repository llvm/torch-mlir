# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.
"""
Example use of the example Torch MLIR LTC backend.
"""
import argparse

import torch.nn.functional as F


def main(device):
    import torch

    if device in ("TS", "MLIR_EXAMPLE"):
        import torch._lazy

        if device == "TS":
            import torch._lazy.ts_backend

            torch._lazy.ts_backend.init()

        elif device == "MLIR_EXAMPLE":
            import ltc_backend.ltc_backend._EXAMPLE_MLIR_BACKEND as ltc_backend

            ltc_backend._initialize()

        device = "lazy"
        print("Initialized backend")
    else:
        device = device.lower()

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
    optimizer.zero_grad()

    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    if device == "lazy":
        print("Calling Mark Step")
        torch._lazy.mark_step()

    print()
    print(loss)


if __name__ == "__main__":
    torch.manual_seed(0)

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
    main(args.device)
