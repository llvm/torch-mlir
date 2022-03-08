1, generate the mnist module .pt format
python train.py
2, run the torch_mlir flow with the generated module.
specify the image url number at line 108 of torchscript_mnist_e2e.py
python torchscript_mnist_e2e.py
3, mnist_torchscript_import.mlir is imported module from torchscript.
   mnist_compiled.mlir is optimized and dumped  