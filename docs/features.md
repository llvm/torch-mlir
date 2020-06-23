# Supported compiler features

## Data types:

The compiler models more datatypes than are implemented by existing backends:

* `bool`
* `bytes`
* `ellipsis`
* `NoneType`
* `str`
* `int` (mapped to either i32 or i64 depending on target)
* `float` (mapped to either f32 or f64 depending on target)

Next steps will extend type support to:

* `tuple`
* `list`
* `dict`
* `range`
* slices

In general, the high level modeling in the `basicpy` dialect will preserve the ability to represent fully dynamically typed forms of containers, but will also support restricted, parametric typed forms suitable for static typed implementations.

### Planned numpy types include:

* Mutable and immutable `ndarray` variants with both known and unknown `dtype` (although most real backends will require a statically inferable dtype).
* Various numpy scalar types that map to low level types already supported by MLIR/LLVM.

## Constants/Literals:

See [the test as the SOT](../pytest/Compiler/constants.py). These will generally follow with data type support above but special notes will be called out here.

## Comparisons:

Full support for comparison ops is expected. See [the test as the SOT](../pytest/Compiler/comparisons.py). Short-circuit control flow is properly emitted for compound comparisons (e.g. `x < y == z >= omega`).

## Binary expressions:

See [the test as the SOT](../pytest/Compiler/binary_expressions.py). All binary expressions should be modeled at the `basicpy` level.

However, for built-in primitives, differences arise at later phases of compilation (some of which are fundamental, and some of which may be eased at a future point). See [primitive_ops_to_std.py](../pytest/Compiler/primitive_ops_to_std.py) for precise lowerings.

Notes:

* NPComp follows the [Numba convention](https://numba.pydata.org/numba-doc/dev/proposals/integer-typing.html) with respect to integer promotion and decisions regarding arbitrary sizes integer values.
* Fully compliant support for div/floor-div modes is not yet supported (see [TODOs in the conversion patterns](../lib/Conversion/BasicpyToStd/PrimitiveOpsConversion.cpp)).

## Logical/Boolean Operations:

* Short-circuiting `and` / `or` operations [are supported](../pytest/Compiler/booleans.py). Note that such operations return the evaluated value, not `bool`, so fewer constraints are available to type inference (as compared to more strongly typed forms of such operations).
* `not` operations
* Conditional (i.e. `1 if True else 2`)

Most of these operations are implemented in terms of the `basicpy.to_boolean` op, which is implemented for built-in types directly by the compiler (see [PrimitiveOpsConversion.cpp](../lib/Conversion/BasicpyToStd/PrimitiveOpsConversion.cpp)).

## Miscellaneous structural components:

See [structure.py](../pytest/Compiler/structure.py).

* `pass` and functions without an explicit return causes the function to return None.
* "Expression statements" are supported but not yet complete/correct. They are currently implemented in terms of the side-effecting `basicpy.exec` op, which wraps an expression and an explicit `basicpy.exec_discard` op to anchor it.

## Name resolution:

How names are resolved can be quite context dependent, varying from only resolving locals all the way to importing globals and maintaining them as mutable. In addition, this intersects with the precise strategy employed to perform "macro expansion" for builtin functions and attribute/index resolution.

Currently, the facility has been set up to be fairly generic (see [environment.py](../python/npcomp/compiler/environment.py)) with a helper for setting up scopes compatibility with global functions that do not form a closure. See:

* [resolve_const.py test](../pytest/Compiler/resolve_const.py)

## Type inference

See [type_inference.py](../pytest/Compiler/type_inference.py).

While transforming from an AST to the `basicpy` dialect, the importer inserted `!basicpy.UnknownType` and corresponding `basicpy.unknown_cast` ops as needed to make the extraction legal. At this phase, if type information is locally known, it is emitted; otherwise, `!basicpy.UnknwonType` is used.

The current [type inference algorithm](../lib/Dialect/Basicpy/Transforms/TypeInference.cpp) is a simple HM-style approach that is just sufficient to do basic propagation as needed to bootstrap (eliminating UnknownType in "simple" cases), but it is not sufficient when considering sub-typing.

Upgrading and fully specifying the type inference behavior is being deferred as possible in favor of getting more of the system bootstrapped, but it will eventually need to be fairly full featured.
