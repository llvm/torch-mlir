# Torch-MLIR Lazy Tensor Core Backend

Lazy Tensor Core support is provided through an abstract [`TorchMlirBackendImpl`](../python/torch_mlir/csrc/base_lazy_backend/backend_impl.h) class. An example implementation is available [here](../examples/ltc_backend/ltc_backend).

There are also examples of a [HuggingFace BERT](../examples/ltc_backend_bert.py) and [MNIST model](../examples/ltc_backend_mnist.py) running on the example/reference LTC backend.
