# Future Work for Lazy Tensor Core

In the last part of the section [Understand The Metrics Report](https://github.com/pytorch/pytorch/blob/lazy_tensor_staging/lazy_tensor_core/TROUBLESHOOTING.md#understand-the-metrics-report), it is mentioned that after running the metrics report,

> If you see `aten::` ops other than `nonzero` and `_local_scalar_dense`, that usually means a missing lowering in the accelerator plugin.

Looking at the sample [output](https://github.com/ramiro050/lazy-tensor-samples/blob/main/lazytensor_resnet18_example_output.txt) and the sample [output](https://github.com/ramiro050/lazy-tensor-samples/blob/main/lazytensor_maskrcnn_example_output.txt) produced by running a [ResNet18](https://github.com/ramiro050/lazy-tensor-samples/blob/main/lazytensor_resnet18_example.py) model and a [MaskRCNN](https://github.com/ramiro050/lazy-tensor-samples/blob/main/lazytensor_maskrcnn_example.py) model, respectively, on the Lazy Tensor Core using the TorchScript backend, the following operations are needed and not yet supported by the backend:

- `aten::convolution_overrideable`
- `aten::max_pool2d_with_indices`
- `aten::mean.out`
- `aten::sort`
- `aten::arange.start_out`
- `aten::bitwise_and.Tensor_out`
- `aten::clamp.out`
- `aten::exp.out`
- `aten::index.Tensor`
- `aten::nonzero`
- `aten::rsqrt.out`
- `aten::sigmoid.out`
- `aten::topk.values`
- `aten::upsample_nearest2d.out`

**Note:** This list is incomplete because currently the MaskRCNN example crashes halfway through when run on LTC. The output error can also be found in the MaskRCNN sample [output](https://github.com/ramiro050/lazy-tensor-samples/blob/main/lazytensor_maskrcnn_example_output.txt).
