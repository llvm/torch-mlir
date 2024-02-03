# TorchOnnx To Torch Conversions

We enable the direct representation of many ONNX features directly in
the `torch` dialect as `torch.operator` custom ops with names like
`onnx.{OperatorName}`. The majority of ONNX operators are represented
with a systematic transformation. `torch_mlir.extras.onnx_importer`
for the reference importer which complies with the rules below.

## Adding new ONNX operators

With the exception of certain special or complicated ONNX operators, most
are relatively straight-forward to map, following this general procedure:

* Plan the ops you wish to support by consulting the
  [ONNX operator database](https://onnx.ai/onnx/operators/).
  * This database has detailed diffs wrt different support versions but
    at the level of detail we operate, most version diffs are inconsequential
    and just require a bit more pattern support.
  * This typically applies to generalization of broadcasting semantics,
    expanded type support, and other things of the like.
* *Prerequisite*: Add support for the op to torch-mlir if it does not
  already exist.
* Open the corresponding implementation file `DefaultDomainXtoY.cpp`
  corresponding with the alphabetic sort of the op and add a conversion.
* Generate successful test cases:
  * All `onnx_importer.py` tests are dumped to the test temp dir (success
    or failure). This is typically located under
    `tools/torch-mlir/test/python/onnx_importer/Output`. The `.mlir` files
    under there should provide good variants to drive lit test coverage of
    conversion.
    * (Optionally) If there is an Onnx file that uses the op of interest,
      convert that file to Onnx MLIR form using the following Python command,
      `python -m torch_mlir.tools.import_onnx my_model.onnx`.
  * There are often many variants of tests for checking conformance of
    different historic ONNX encodings, but these are often not load bearing
    at the MLIR level.
  * Pick a handful of test cases and add them to
    `test/Conversion/TorchOnnxToTorch/simple_ops_x_to_y.mlir` corresponding to
    an alphabetic breakdown. At this time, ignore tests that are not exercising
    useful differences in the pattern implementations.
    * (Optionally) Use `torch-mlir-opt` to validate the outputs of the new op.
      First, build the project using
      `cmake --build build --target tools/torch-mlir/all`. This will generate
      the conversion binary, `torch-mlir-opt`. Then call `torch-mlir-opt` with
      the MLIR pass `convert-torch-onnx-to-torch`:
      ```
      build/bin/torch-mlir-opt -convert-torch-onnx-to-torch \
      -split-input-file [DESIRED_ONNX_FILE].mlir
      ```
* Generate failure test cases:
  * Some ops have forms that do not (easily) map to torch-mlir. If you leave
    an op under-implemented, add a failing test case to
    `test/Conversion/TorchOnnxToTorch/unsupported_simple_ops.mlir`.
* Optional but recommended: Use your test case files to fuzz against the
  torch-mlir backend of your choice by running a backend conversion pipeline
  and fixing any crashes/issues.
* Send a patch with your changes.

## ONNX proto to `torch` dialect mapping

### Type Conversion

* Tensors: ONNX tensor types are converted to `torch.vtensor`
  with static and dynamic dimensions. We require that shape
  inference has run to produce ranked tensors.
* Tensor element types are directly converted to corresponding
  MLIR types as used by the rest of torch-mlir.
* String, sequence and sparse tensor types are presently not mapped.

### Attributes

A subset of attributes types are converted directly to an attribute
dict on the op with a name like `torch.onnx.{AttributeName}`. The
following attribute type mappings are made:

* `FLOAT`: `FloatAttr`
* `INT`: Signed `IntegerAttr` of width 64
* `STRING`: `StringAttr`
* `TENSOR`: Converted to one of:
  * `DenseResourceElementsAttr` for inlined `raw_data`
  * `DenseElementsAttr` for splats
  * `DenseElementsAttr` for inlined typed proto initialization
* `FLOATS`: `ArrayAttr` of `FloatAttr`
* `INTS`: `ArrayAttr` of signed `IntegerAttr` of width 64
* `STRINGS`: `ArrayAttr` of `StringAttr`
* `TENSORS`: `ArrayAttr` of corresponding `TENSOR` conversion

The following attribute types have no present, systematic conversion.
Their presence on an op indicates that the op is a special form, which
must be handled specially:

* `GRAPH`
* `SPARSE_TENSOR` (TBD: it is possible to handle this systematically if
  useful).
* `TYPE_PROTO` (TBD: it may be possible to handle this systematically if
  useful).
* Plural equivalents of the above.

### Default operation conversion

Operations are converted to a `torch.operator` with name `onnx.{OperatorName}`.
The constraint that the ONNX graph is topologically sorted and free of
cycles matches the SSA form. Operands and results are mapped directly.

This conversion only applies to the default (empty) domain.

### Quantization information

Quantization parameters are carried out of line in the ONNX protobuf
and will be repatriated upon import to torch. The exact mechanism is
not yet implemented.

### Version and metadata

The `IsolatedFromAbove` parent of the ops can contain the following
metadata:

* `torch.onnx_meta.ir_version`: 64bit `IntegerAttr` corresponding to
  `ModelProto.ir_version`.
* `torch.onnx_meta.producer_name`: `StringAttr` corresponding to
  `ModelProto.producer_name`.
* `torch.onnx_meta.producer_version`: `StringAttr` corresponding to
  `ModelProto.producer_version`.
* `torch.onnx_meta.opset_version`: 64bit `IntegerAttr` corresponding
  to `ModelProto.opset_import.version` for the domain "" (empty).
  Will be ommitted if the default opset is not included.
* `torch.onnx_meta.opset_versions`: DictAttr of 64bit `IntegerAttr`
  for each non default domain.

Generally, the importer handles variations in `ir_version` whereas
the transformations here handle opset version differences. Version
independent transformations are encouraged where possible if there
are only minor variations of an op. Major variations should use
`since_version` sensitive patterns.

### Special op forms

Certain ONNX operators map to different structural components of
torch-mlir's representation:

* `ConstantOfShape`: Mapped to `torch.vtensor.literal` with
  a corresponding `value` attribute.

