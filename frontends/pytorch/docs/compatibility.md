# Compatibility notes

This document contains known compatibility issues with the PyTorch integration.
Some items may be permanent limitations and other may just be capturing
plans for future work. In general, this only applies to the default
configuration, not the "type dispatch" (PyTorch 1.4) integration.


## Pending Work Needed

### Unpacking quantized weights

Some of the torch::jit::Operator 's (especially the quantized::  ones) have already gone through some lowering steps. Specifically, the quantized::conv operators are pre-packed and stored as ConvPackedParam attributes on the Module instead of just passing the weight/bias tensors as SSA arguments to the Operator[0] [1]. How they are packed depends on whether the fbgemm or the XNNPACK backends are used...

I think this comes back to the ability to pass a "CustomClass" as an SSA Value into an Operator, which may be difficult for us to lower to TCF...
Glow (and others) get around this by adding custom passes to convert the PackedParams to a traditional glow::unpacked_quantized_conv operation [2], but that adds some layers of lowering in TorchScript land before we would want to call off to get_registered_ops on the python side (may not be avoidable?)

[0]: https://github.com/pytorch/pytorch/blob/dc67b47bc9d53dbeb898a4d920b0225ac73629ec/aten/src/ATen/native/quantized/library.cpp#L63-L69
[1]: https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/quantized/cpu/conv_packed_params.h
[2]: https://github.com/pytorch/glow/blob/b62ec449c43b77722c119b53b3ea5aec9be3edb9/torch_glow/src/TorchGlowBackend.cpp#L98-L116
