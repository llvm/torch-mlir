# mnist-playground

This is intended to be a short-lived "playground" for doing various experiments, guided by a real model use case, for improving the npcomp reference backend.

It's expected that utilities developed here will graduate to a more general utility or that this utility will be obsoleted by Python-driven flows once those come online.

## Goals:

- Obtain a performance-grounded analysis of the TCF/TCP design + reference backend design, and improve the designs.

- Make forward progress on TCF/TCP + reference backend while the PyTorch frontend is being brought up.

## Rough sketch of how we intend to get there:

1. Link against PyTorch, and write a simple routine to do inference on a simple FC MNIST.

2. Write a similar routine in TCF, extending TCF and the reference backend as needed for functional completeness. The PyTorch code serves as a numerical correctness reference.

3. Run and profile the reference backend and obtain a set of action items for design improvements, both to performance and stability. The PyTorch code serves as a performance baseline.

4. Implement important action items on a priority basis, and document remaining major design issues that don't make sense to address at this time, along with a justification for why the current design doesn't prevent us from eventually solving them. Iterate the previous step and this one as makes sense.

5. (Stretch) Add support for convolutional MNIST and/or training.

## Current Status

Step 1. DONE

Step 2. MOSTLY DONE. Still need to improve the op set to make the FC MNIST more complete. In particular, implementing functionality for reshape and softmax.

Step 3. STARTING. Initial performance on 10x784x100 (10 FC feature, batch 100) is 66x off from PyTorch. No profiling done yet.

Example command line (the .mlir file and `-invoke` are similar to npcomp-run-mlir):

```
$ mnist-playground tools/mnist-playground/fc.mlir -invoke fc
PyTorch: numRuns: 16384 nsPerRun: 3.947563e+05
RefBackend: numRuns: 256 nsPerRun: 2.471073e+07
Ratio (RefBackend / PyTorch): 62.5974
```

There is currently a fragile dependency between hardcoded `at::` function calls in the .cpp file and the TCF code in the `.mlir` file. A correctness check is done to make sure they agree. Once we have a PyTorch frontend and/or ATen roundrip ATen backend oneline, we can avoid this fragility.
