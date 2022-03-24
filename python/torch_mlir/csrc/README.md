#Torch - MLIR Lazy Tensor Core Backend

Contained within this directory are the components that implements the
Torch-MLIR LTC backend.

The components are subclasses of the backend API interface classes found under
[torch/csrc/lazy/backend](https://github.com/pytorch/pytorch/tree/master/torch/csrc/lazy/backend).

Importantly, the subclasses are still abstract classes. Pure virtual methods
such as `Compile` were purposefully not overriden as Torch-MLIR does not know
how to compile the model for the target hardware.

The intent is that vendor hardware specific plugins will subclass the Torch-MLIR
backend classes and override the remaining pure virtual functions to complete
the backend.

The Torch-MLIR LTC backend's job is to perform the lowering from ATen to MLIR. A
hardware vendor's backend job is to take care of the actual compile and
execution of the lowered MLIR.