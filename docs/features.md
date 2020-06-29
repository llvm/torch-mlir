# Supported compiler features

## Basic python support

At the core, the compiler models a subset of the python language with various
ways to extend this to provide compilation support for custom libraries, etc.

### Data types:

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

#### Planned numpy types include:

* Mutable and immutable `ndarray` variants with both known and unknown `dtype` (although most real backends will require a statically inferable dtype).
* Various numpy scalar types that map to low level types already supported by MLIR/LLVM.

### Constants/Literals:

See [the test as the SOT](../pytest/Compiler/constants.py). These will generally follow with data type support above but special notes will be called out here.

### Comparisons:

Full support for comparison ops is expected. See [the test as the SOT](../pytest/Compiler/comparisons.py). Short-circuit control flow is properly emitted for compound comparisons (e.g. `x < y == z >= omega`).

### Binary expressions:

See [the test as the SOT](../pytest/Compiler/binary_expressions.py). All binary expressions should be modeled at the `basicpy` level.

However, for built-in primitives, differences arise at later phases of compilation (some of which are fundamental, and some of which may be eased at a future point). See [primitive_ops_to_std.py](../pytest/Compiler/primitive_ops_to_std.py) for precise lowerings.

Notes:

* NPComp follows the [Numba convention](https://numba.pydata.org/numba-doc/dev/proposals/integer-typing.html) with respect to integer promotion and decisions regarding arbitrary sizes integer values.
* Fully compliant support for div/floor-div modes is not yet supported (see [TODOs in the conversion patterns](../lib/Conversion/BasicpyToStd/PrimitiveOpsConversion.cpp)).

### Logical/Boolean Operations:

* Short-circuiting `and` / `or` operations [are supported](../pytest/Compiler/booleans.py). Note that such operations return the evaluated value, not `bool`, so fewer constraints are available to type inference (as compared to more strongly typed forms of such operations).
* `not` operations
* Conditional (i.e. `1 if True else 2`)

Most of these operations are implemented in terms of the `basicpy.to_boolean` op, which is implemented for built-in types directly by the compiler (see [PrimitiveOpsConversion.cpp](../lib/Conversion/BasicpyToStd/PrimitiveOpsConversion.cpp)).

### Miscellaneous structural components:

See [structure.py](../pytest/Compiler/structure.py).

* `pass` and functions without an explicit return causes the function to return None.
* "Expression statements" are supported but not yet complete/correct. They are currently implemented in terms of the side-effecting `basicpy.exec` op, which wraps an expression and an explicit `basicpy.exec_discard` op to anchor it.

### Name resolution:

How names are resolved can be quite context dependent, varying from only resolving locals all the way to importing globals and maintaining them as mutable. In addition, this intersects with the precise strategy employed to perform "partial evaluation" for builtin functions and attribute/index resolution.

Currently, the facility has been set up to be fairly generic (see [environment.py](../python/npcomp/compiler/environment.py)) with a helper for setting up scopes compatibility with global functions that do not form a closure. See:

* [resolve_const.py test](../pytest/Compiler/resolve_const.py)

### Type inference

See [type_inference.py](../pytest/Compiler/type_inference.py).

While transforming from an AST to the `basicpy` dialect, the importer inserted `!basicpy.UnknownType` and corresponding `basicpy.unknown_cast` ops as needed to make the extraction legal. At this phase, if type information is locally known, it is emitted; otherwise, `!basicpy.UnknwonType` is used.

The current [type inference algorithm](../lib/Dialect/Basicpy/Transforms/TypeInference.cpp) is a simple HM-style approach that is just sufficient to do basic propagation as needed to bootstrap (eliminating UnknownType in "simple" cases), but it is not sufficient when considering sub-typing.

Upgrading and fully specifying the type inference behavior is being deferred as possible in favor of getting more of the system bootstrapped, but it will eventually need to be fairly full featured.

### Partial evaluation

Out of a desire to extract programs from a running python session, a facility
exists to perform partial evaluation of key operations against live python
values referenced from an outer environment. Actual use of this facility
yields a language that is "not python" anymore, but if sufficiently well
defined, the argument is that it can still be intuitive. The facility is
completely opt-in, based on passing a `PartialEvalHook` to the `Environment` used
to compile a function. The `PartialEvalHook` can bind logic to:

* Specific references (i.e. functions like `len`)
* Types (checked via issubclass)
* Arbitrary lambda predicates

When evaluating names against certain scopes that contain live values, the importer pre-processes the live value through the `PartialEvalHook`, either letting it generate:

* A `LiveValueRef` that defines further allowed partial evaluation operations on the value.
* A materialized IR value
* An error

Further, evaluation of expressions containing such partial evaluation results is deferred to a special AST visitor that will attempt to match partial evaluation invocations prior to emitting the corresponding code. `LiveValueRef` can provide import time special processing for:

* getattr
* call (not yet implemented)
* index (not yet implemented)

In this way, a DSL can be constructed that effectively subsets the parts of the python environment that are supported. As an example, there is a default `PartialEvalHook` used for testing that enables `getattr` resolution against modules and tuples (including namedtuple). Combined with a `ConstModuleNameResolver` for resolving global names as constants, this allows code like the following to compile:

```python
# CHECK-LABEL: func @module_constant
@import_global
def module_constant():
  # CHECK: constant 3.1415926535897931 : f64
  return math.pi


Sub = collections.namedtuple("Sub", "term")
Record = collections.namedtuple("Record", "fielda,fieldb,inner")
record = Record(5, 25, Sub(6))


# CHECK-LABEL: func @namedtuple_attributes
@import_global
def namedtuple_attributes():
  # CHECK: constant 6
  # CHECK: constant 25
  return record.inner.term - record.fieldb
```

This is accomplished with the following `PartialEvalHook` setup:

```python
  mr = PartialEvalHook()
  #### Modules
  mr.enable_getattr(for_type=ast.__class__)  # The module we use is arbitrary.

  #### Tuples
  # Enable attribute resolution on tuple, which includes namedtuple (which is
  # really what we want).
  mr.enable_getattr(for_type=tuple)
  return mr
```

It is expected that this facility will evolve substantially, as it is the primary intended mechanism for remapping significant parts of the python namespace to builtin constructs (i.e. it will be the primary way to map `numpy` functions and values).

### Calls

This is very much a WIP. Relevant ops:

* `func_template`: Aggregates a list of function overloads to choose for a symbolic name.
* `func_template_call`: Performs a symbolic call with python source conventions.

The idea is that a library modules of `func_template` definitions is assembled with all concrete implementations that have compiler support. The python compiler will iterate over all such templates and bind partial evaluation rules in the environment to detect the calls. Then, when importing, `func_template_call` ops make the call.

See the `basicpy.func_template` op for more detailed notes. The intention is that compiler-supported functions, methods, attribute getter/setter, and dunder functions can all exist in the library, with concrete resolution carried out by type constraints and corresponding type inference passes. Upon matches, concrete functions are pulled into the module being compiled and possibly inlined. With enough type constraints and some iteration, this should converge reasonably to a statically typed program.

See the tests:

* [template_call.py](../pytest/Compiler/template_call.py)


## Numpy extension

Numpy compilation is factored as
[an extension](../python/npcomp/compiler/extensions/numpy) of the core python compiler, using the following features:

* Value coders for importing constant ndarrays.
* Partial evaluation hooks for emitting IR for various built-in ops.
* Integration with the type inference system (future).
* Integration with the library call mechanism in order to resolve `ndarray`
  methods and non-op parts of the API (future).

The [`numpy` dialect](../include/npcomp/Dialect/Numpy/IR/NumpyOps.td) is logically an extension of the [`basicpy` dialect](../include/npcomp/Dialect/Basicpy/IR/BasicpyOps.td), extending type support and providing coverage for numpy built-ins.

### Tensor and array types

A goal of the numpy extension is to enable modeling of a strict subset of the numpy API, relying on compiler tooling to bridge the gap to runtime systems that (potentially) take different opinions. As such, it has to model core types such as `ndarray` in a way that preserves semantics. The semantic that tends to give the most trouble is mutability: it is common for compilers to prefer numeric programs in value form; however, we have to meet numpy where it is. History has shown that prematurely restricting the highest levels to immutability, leaks concepts and "weirdness" in various ways that are hard to patch up later (this is a controversial statement).

Since a large portion of the numpy op surface area does not alias buffers, and those that do can (typically) be structurally identified, this extension chooses to err on the side of making a mutable `ndarray` be the basic datatype but ops that do not alias are defined in terms of the built-in `tensor` type (which represents an immutable value). Operations exist to copy an `ndarray` to a `tensor` and create a new `ndarray` from a tensor in order to bridge the worlds.

For programs that do not introduce aliasing, it is within the bounds of simple canonicalizations or other local transforms to elide such copies and reduce to value types. For more complicated programs that do rely on aliasing, more complicated analysis/transforms can be done -- and which are are needed is somewhat backend specific (since some backends define themselves in terms of pure-value semantics whereas others carry a concept of buffer mutability all the way to their frontend).

As an example of the kind of IR this produces, consider this (non-canonicalized) import of the simple python function:

```python
a = np.asarray([1.0, 2.0])
b = np.asarray([3.0, 4.0])

@import_global
def global_add():
  return np.add(a, b)
```

```mlir
  func @global_add() -> !basicpy.UnknownType attributes {iree.module.export} {
    %cst = constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf64>
    %0 = numpy.create_array_from_tensor %cst : (tensor<2xf64>) -> !numpy.ndarray<f64>
    %cst_0 = constant dense<[3.000000e+00, 4.000000e+00]> : tensor<2xf64>
    %1 = numpy.create_array_from_tensor %cst_0 : (tensor<2xf64>) -> !numpy.ndarray<f64>
    %2 = numpy.copy_to_tensor %0 : (!numpy.ndarray<f64>) -> tensor<*xf64>
    %3 = numpy.copy_to_tensor %1 : (!numpy.ndarray<f64>) -> tensor<*xf64>
    %4 = numpy.builtin_ufunc_call<"numpy.add"> (%2, %3) : (tensor<*xf64>, tensor<*xf64>) -> tensor<*x!basicpy.UnknownType>
    %5 = numpy.create_array_from_tensor %4 : (tensor<*x!basicpy.UnknownType>) -> !numpy.ndarray<?>
    %6 = basicpy.unknown_cast %5 : !numpy.ndarray<?> -> !basicpy.UnknownType
    return %6 : !basicpy.UnknownType
  }
```

### Ufuncs

So called [universal functions](https://numpy.org/doc/stable/reference/ufuncs.html) are a core numpy abstraction covering a variety of applications of elementwise operations. Presently, the compiler imports all `__call__` operations on them as a `numpy.builtin_ufunc_call` op with a symbolic name. This facility is expected to be extended later.

In addition, other ufunc application functions still need to be defined:

* `reduce`
* `accumulate`
* `reduceat`
* `outer`
* `at`

It is expected that in addition to the black-box "builtin" ufuncs, the facility will grow to include library call based variants which can define their own scalar form.
