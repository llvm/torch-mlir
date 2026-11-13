# User annotations (`mlir.user`)

Torch-MLIR lets users attach arbitrary metadata to values in a PyTorch program
and carries that metadata through lowering, so a downstream compiler can read it
in backend IR (linalg-on-tensors, TOSA, StableHLO). For example, a compiler may
need range estimates for the inputs of activation functions.

## Producing annotations

Wrap a tensor with `torch_mlir.extras.annotate.annotate_and_pass_through`
(alias `AnnotateAndPassThrough`). This identity op takes a dictionary of
annotations:

```python
from torch_mlir.extras.annotate import AnnotateAndPassThrough

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 4)

    def forward(self, x):
        x = AnnotateAndPassThrough(x, {"foo.bar": "baz"})
        y = self.fc(x)
        y = AnnotateAndPassThrough(
            y, {"domain.lower_bound": -1.0, "domain.upper_bound": 1.0}
        )
        return torch.relu(y)
```

The FX importer does not emit the op itself. Instead, it attaches the dictionary
to the value the op wraps. Python values become attributes as follows: `int` →
`i64`, `float` → `f64`, `str` → `StringAttr`, `list`/`tuple` → `ArrayAttr`,
`True` → `UnitAttr` (`False` is omitted). A pre-built `mlir.ir.Attribute` is
used unchanged.

## Attribute structure

All user annotations live under a single reserved attribute name, `mlir.user`.
Keys are never lifted to the top level, so they cannot collide with inherent or
dialect attributes. The contents of each dictionary are opaque to Torch-MLIR.

* **Op results.** `mlir.user` is a discardable `ArrayAttr` of
  `DictionaryAttr`, with one entry per op result, indexed by result number.
  Results without annotations hold an empty dictionary.

  ```mlir
  %values, %indices = torch.aten.topk %x, %k, %dim, %true, %true
      {mlir.user = [{tag = "values", lo = -1.0 : f64}, {tag = "indices"}]} : ...
  ```

* **Function arguments.** `mlir.user` is a `DictionaryAttr` argument
  attribute, since an argument denotes exactly one value.

  ```mlir
  func.func @main(%arg0: !torch.vtensor<[1,4],f32> {mlir.user = {foo.bar = "baz"}})
  ```

## Propagation

Lowering replaces ops many times, through decomposition, canonicalization and
conversion to a backend dialect. Each time an annotated op is replaced, the
dictionary for result *i* moves to the producer of the value that replaces
result *i*. After a chain of replacements, the annotation therefore ends up on
the **last op in the chain**: the op that produces the final value standing in
for the original result.

For the example above, `nn.Linear` decomposes to `mm` + `add`, and the
annotation on its output lands on the `add`. Lowering to linalg-on-tensors then
replaces the `add` with a `linalg.generic`, which receives the annotation:

```mlir
func.func @main(%arg0: tensor<1x4xf32> {mlir.user = {foo.bar = "baz"}}) -> tensor<1x4xf32> {
  ...
  %3 = linalg.matmul ins(%arg0, %transposed : ...) outs(%2 : ...) -> tensor<1x4xf32>
  %5 = linalg.generic {...} ins(%3, %4 : ...) outs(%1 : ...)
      attrs =  {mlir.user = [{domain.lower_bound = -1.000000e+00 : f64, domain.upper_bound = 1.000000e+00 : f64}]} {
    ...  // arith.addf
  } -> tensor<1x4xf32>
  %6 = linalg.generic {...} ins(%5 : ...) outs(%1 : ...) {
    ...  // relu
  } -> tensor<1x4xf32>
  return %6 : tensor<1x4xf32>
}
```

The precise rules:

* **Only the replacement value is annotated.** Helper ops created along the way
  (the `matmul`, broadcasts, shape computations) get nothing. Ops that consume
  the annotated value (the `relu` above) do not inherit the annotation either.
* **Casts are looked through.** If the replacement value comes from
  `tensor.cast` or a single-input `unrealized_conversion_cast`, the annotation
  goes on the op that produces the cast's input.
* **Replacement by a function argument.** If the replacement value is a
  function entry-block argument (for example, when an identity op folds away),
  the dictionary is merged into that argument's `mlir.user` attribute.
* **Merging.** If the target already carries a dictionary for that result or
  argument, the dictionaries are merged, and keys already present on the target
  win. A downstream fold never overwrites an annotation placed upstream.
* **Function arguments** keep their `mlir.user` dictionary through signature
  and type conversion.
* **Folded and dead ops.** If an annotated op folds to a constant or is erased
  because it is dead, there is no replacement value, and the annotation is
  dropped.

## Testing

End-to-end tests of each backend pipeline provide best-effort coverage.
For example (as of [#4627])(https://github.com/llvm/torch-mlir/pull/4627))
these tests check each pipeline:

* `test/Conversion/TorchToLinalg/backend-pipeline-user-attrs.mlir`
* `test/Conversion/TorchToTosa/backend-pipeline-user-attrs.mlir`
* `test/Conversion/TorchToStablehlo/backend-pipeline-user-attrs.mlir`
* `test/python/fx_importer/annotate_*_test.py` (FX import through to
  linalg-on-tensors)

Known gaps can occur due to out-of-tree passes that don't preserve attributes,
and are recorded as `XFAIL` tests.

## Writing passes that preserve annotations

Passes in these pipelines forward annotations automatically when they use the
helpers in `torch-mlir/Conversion/Utils/Utils.h`:

```c++
RewritePatternSet patterns(context);
// ... populate patterns ...
wrapPatternsWithForwarding(patterns);
auto listener = createConversionForwardingListener();

// Greedy rewrites:
GreedyRewriteConfig config;
config.setListener(listener.get());
(void)applyPatternsGreedily(getOperation(), std::move(patterns), config);

// Dialect conversion:
ConversionConfig config;
config.listener = listener.get();
(void)applyPartialConversion(getOperation(), target, std::move(patterns), config);
```

Use both helpers. In greedy rewrites, the listener sees every replacement as it
happens. In dialect conversion, replacements are only committed at the end, so
the pattern wrapper forwards annotations as soon as each pattern succeeds;
otherwise annotations on intermediate ops would be lost.

Passes that replace ops without a pattern driver can call
`forwardUserAttrs(oldOp, replacementValues)` directly.
