#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections import namedtuple
from enum import Enum

import numpy as np

__all__ = [
    "Unspec",
    "ArrayConstraint",
    "ArrayParams",
    "DType",
    "DimFlag",
    "DimFlagEnum",
    "DynamicDim",
    "Rank",
    "Shape",
    "Signature",
    "TypeClass",
    "TypeConstraints",
    "ValueType",
]

# TODO: All supported types
_DTYPE_TO_ASM_DICT = {
    np.bool: "i1",  # TODO: May need a custom type to signify 8bit storage
    np.int8: "s8",
    np.int16: "s16",
    np.int32: "s32",
    np.int64: "s64",
    np.float32: "f32",
    np.float64: "f64",
}


def _dtype_to_mlir_asm(dtype):
  return _DTYPE_TO_ASM_DICT.get(dtype)


class _LiterateEnum(Enum):
  """An enum that can be parsed/printed based on its name.

    >>> class SampleEnum(_LiterateEnum):
    ...   Red = 1
    ...   Blue = 2
    >>> SampleEnum.Red
    Red
    >>> SampleEnum.parse("Red")
    Red
    >>> SampleEnum.parse("Mauve")
    Traceback (most recent call last):
    ...
    ValueError: Cannot parse SampleEnum 'Mauve'
    >>> SampleEnum.parse("parse")
    Traceback (most recent call last):
    ...
    ValueError: Cannot parse SampleEnum 'parse'
    >>> SampleEnum.parse(None)
    Traceback (most recent call last):
    ...
    ValueError: Cannot parse SampleEnum None
    >>> SampleEnum.parse(1.0)
    Traceback (most recent call last):
    ...
    ValueError: Cannot parse SampleEnum 1.0

  """

  @classmethod
  def parse(cls, v):
    if isinstance(v, cls):
      return v
    if not v or not isinstance(v, str) or v[0] == '_' or not hasattr(cls, v):
      raise ValueError("Cannot parse %s %r" % (
          cls.__name__.split(".")[-1],
          v,
      ))
    value = getattr(cls, v)
    if not isinstance(value, cls):
      raise ValueError("Cannot parse %s %r" % (
          cls.__name__.split(".")[-1],
          v,
      ))
    return value

  def __repr__(self):
    return self.name


# Special "unspecified" value that we use throughout.
class _Unspec:
  __slots__ = []

  def __str__(self):
    return "Unspec"

  def __repr__(self):
    return "Unspec"


Unspec = _Unspec()


class TypeClass(_LiterateEnum):
  """Top level types in the npcomp language."""
  Any = 0
  NdArray = 1


class ValueType:
  """The type a value can take in the npcomp language.

  Types of values in npcomp are always being refined and are therefore
  mutable. Instances represent the type derived for a single value, not a
  concept of "typeness" generally.

    >>> ValueType()
    Any
    >>> ValueType('NdArray')
    NdArray
    >>> ValueType('NdArray', DType(np.float32), Rank(2))
    NdArray[DType(float32), Rank(2)]
    >>> vt = ValueType('NdArray')
    >>> vt += Rank(3)
    >>> vt += DynamicDim(1)
    >>> vt
    NdArray[Rank(3), DimFlag(Dynamic, (1,))]
    >>> vt = ValueType()
    >>> vt.type_class = 'NdArray'
    >>> vt
    NdArray
  """
  __slots__ = ["_constraints", "_type_class"]

  def __init__(self, type_class=TypeClass.Any, *constraints):
    super().__init__()
    self._type_class = TypeClass.parse(type_class)
    self._constraints = TypeConstraints(constraints)

  def __iadd__(self, constraint):
    assert isinstance(
        constraint, TypeConstraint), ("Can only add constraints to a ValueType")
    self._constraints.append(constraint)
    return self

  def __repr__(self):
    if not self._constraints:
      return repr(self._type_class)
    return "%r[%s]" % (self._type_class, ", ".join(
        [repr(c) for c in self._constraints]))

  @property
  def type_class(self):
    return self._type_class

  @type_class.setter
  def type_class(self, type_class):
    self._type_class = TypeClass.parse(type_class)

  @property
  def constraints(self):
    return self._constraints


class ValueTypeList:
  """Models a list of ValueTypes.

    >>> v3 = ValueTypeList(3)
    >>> v3
    (Any, Any, Any)
    >>> v3[1]
    Any
    >>> v3[2] = 'NdArray'
    >>> v3
    (Any, Any, NdArray)
    >>> v3[2] += Rank(2)
    >>> v3
    (Any, Any, NdArray[Rank(2)])

  With names:
    >>> v3 = ValueTypeList(3, [None, "b", None])
    >>> v3[1] = 'NdArray'
    >>> v3["b"]
    NdArray
    >>> v3["b"] = 'Any'
    >>> v3
    (Any, Any, Any)
  """
  __slots__ = ["_list", "_names"]

  def __init__(self, arity=0, names=None):
    self._list = [ValueType() for _ in range(arity)]
    self._names = names

  def _key_to_index(self, key):
    if isinstance(key, str):
      # Scan for the index.
      if self._names:
        for i, n in enumerate(self._names):
          if n == key:
            return i
      raise KeyError("Unknown key '%s'" % key)
    return key

  def __getitem__(self, key):
    return self._list[self._key_to_index(key)]

  def __setitem__(self, key, value):
    if not isinstance(value, ValueType):
      value = ValueType(value)
    self._list[self._key_to_index(key)] = value

  def __iter__(self):
    return self._list.__iter__()

  def __repr__(self):
    return "(%s)" % (", ".join(repr(t) for t in self._list),)


class Signature:
  """A function signature.

    This currently only models a linear list of positional arguments and
    assumes that multiple results will be represented by some form of tuple
    type.

      >>> Signature()
      () -> Any
      >>> Signature(2)
      (Any, Any) -> Any
      >>> s = Signature(2)
      >>> s.args[1] = 'NdArray'
      >>> s.args[1] += Rank(2)
      >>> s
      (Any, NdArray[Rank(2)]) -> Any
      >>> s.result = 'NdArray'
      >>> s.result += Rank(3)
      >>> s
      (Any, NdArray[Rank(2)]) -> NdArray[Rank(3)]
      >>> s.arg_names[0] = 'a'
      >>> s.arg_names[1] = 'b'
      >>> s
      (a: Any, b: NdArray[Rank(2)]) -> NdArray[Rank(3)]
  """
  __slots__ = ["_args", "_arg_names", "_result"]

  def __init__(self, arity=0):
    super().__init__()
    self._result = ValueType()
    self._arg_names = [None] * arity
    self._args = ValueTypeList(arity, names=self._arg_names)

  @property
  def args(self):
    return self._args

  @property
  def arg_names(self):
    return self._arg_names

  @property
  def result(self):
    return self._result

  @result.setter
  def result(self, value):
    if not isinstance(value, ValueType):
      value = ValueType(value)
    self._result = value

  def __repr__(self):
    args_repr = "(%s)" % (", ".join(
        ((n + ": " + repr(t)) if n else repr(t))
        for t, n in zip(self._args, self._arg_names)),)
    return "%s -> %r" % (args_repr, self._result)


class ArrayParams:
  """Represents parameters defining how to construct an array.

    >>> ArrayParams()
    ArrayParams(dtype=Unspec)
    >>> ArrayParams(np.float32)
    ArrayParams(dtype=float32)
    >>> ArrayParams(np.float32, rank=4)
    ArrayParams(dtype=float32, shape=(-1, -1, -1, -1))
    >>> ArrayParams(np.float32, shape=(1, 2, 3))
    ArrayParams(dtype=float32, shape=(1, 2, 3))
  """
  __slots__ = ["dtype", "shape"]

  def __init__(self, dtype=Unspec, shape=Unspec, rank=Unspec):
    self.dtype = dtype
    if shape is not Unspec:
      self.shape = shape
    elif rank is not Unspec:
      self.shape = [-1 for _ in range(rank)]
    else:
      self.shape = Unspec

  @property
  def rank(self):
    if self.shape is Unspec:
      return Unspec
    return len(self.shape)

  @classmethod
  def from_constraints(cls, constraints):
    """Constructs params for a TypeConstraints list.

    Unconstrained:
      >>> ArrayParams.from_constraints(TypeConstraints())
      ArrayParams(dtype=Unspec)

    DType constrained:
      >>> ArrayParams.from_constraints(TypeConstraints(DType(np.float32)))
      ArrayParams(dtype=float32)

    Rank constrained:
      >>> ArrayParams.from_constraints(TypeConstraints(Rank(2)))
      ArrayParams(dtype=Unspec, shape=(-1, -1))

    Shape constrained:
      >>> ArrayParams.from_constraints(TypeConstraints(Shape(1, 2, 3)))
      ArrayParams(dtype=Unspec, shape=(1, 2, 3))
      >>> ArrayParams.from_constraints(TypeConstraints(
      ...   Rank(3), Shape(1, 2, 3)))
      ArrayParams(dtype=Unspec, shape=(1, 2, 3))

    Shape constrained with dynamic dim constraint:
      >>> ArrayParams.from_constraints(TypeConstraints(
      ...   Shape(1, 2, 3), DynamicDim(1)))
      ArrayParams(dtype=Unspec, shape=(1, -1, 3))
      >>> ArrayParams.from_constraints(TypeConstraints(
      ...   Shape(1, 2, 3), DynamicDim((0, 2))))
      ArrayParams(dtype=Unspec, shape=(-1, 2, -1))

    Errors:
      >>> ArrayParams.from_constraints(TypeConstraints(
      ...   Rank(4), Shape(1, 2, 3)))
      Traceback (most recent call last):
      ...
      ValueError: Conflicting shape and rank: Rank(4) vs Shape(1, 2, 3)
      >>> ArrayParams.from_constraints(TypeConstraints(
      ...   Shape(1, 2, 3), DynamicDim((0, 5))))
      Traceback (most recent call last):
      ...
      ValueError: Out of range DimFlag(Dynamic, (0, 5)) for shape [-1, 2, 3]
    """
    # TODO: Should have a 'canonicalize' method on TypeConstraints which
    # reduces and verifies.
    dtype_c = constraints.one_of(DType)
    shape_c = constraints.one_of(Shape)
    rank_c = constraints.one_of(Rank)
    dim_flags = constraints.all_of(DimFlag)

    dtype = dtype_c.dtype if dtype_c else Unspec
    shape = Unspec

    # Compute shape
    if shape_c:
      # TODO: Should be in canonicalizer
      if rank_c and rank_c.rank != len(shape_c.dims):
        raise ValueError("Conflicting shape and rank: %r vs %r" %
                         (rank_c, shape_c))
      shape = list(shape_c.dims)
    elif rank_c:
      shape = [-1 for _ in range(rank_c.rank)]

    # Apply dim flags
    if shape is not Unspec and dim_flags:
      for df in dim_flags:
        flag, for_dims = df.dim_flag
        for d in for_dims:
          if d < 0 or d >= len(shape):
            raise ValueError("Out of range %r for shape %r" % (df, shape))
          if flag == DimFlagEnum.Dynamic:
            shape[d] = -1

    return cls(dtype=dtype, shape=shape)

  def __repr__(self):
    try:
      s = "ArrayParams(dtype=%s" % (self.dtype.__name__ if isinstance(
          self.dtype, type) else self.dtype,)
      if self.shape is not Unspec:
        s += ", shape=%r" % (tuple(self.shape),)
      s += ")"
      return s
    except:
      return "ArrayParams(ERROR)"

  @property
  def is_concrete(self):
    """Returns true if the parameters are sufficient to construct an ndarray.

      >>> ArrayParams().is_concrete
      False
      >>> ArrayParams(dtype=np.float32).is_concrete
      False
      >>> ArrayParams(dtype=np.float32, rank=1).is_concrete
      False
      >>> ArrayParams(dtype=np.float32, shape=(1, 2)).is_concrete
      True
    """
    if self.dtype is Unspec:
      return False
    if self.shape is Unspec:
      return False
    if any(d < 0 for d in self.shape):
      return False
    return True

  @property
  def mlir_tensor_type_asm(self):
    """Get a corresponding MLIR tensor type.

    Fully Unspecified:
      >>> ArrayParams().mlir_tensor_type_asm
      'tensor<*x!numpy.any_dtype>'

    Unranked:
      >>> ArrayParams(dtype=np.float32).mlir_tensor_type_asm
      'tensor<*xf32>'

    Ranked:
      >>> ArrayParams(dtype=np.float32, rank=3).mlir_tensor_type_asm
      'tensor<?x?x?xf32>'
      >>> ArrayParams(dtype=np.float32, shape=(-1, -1)).mlir_tensor_type_asm
      'tensor<?x?xf32>'

    Scalar:
      >>> ArrayParams(dtype=np.float32, rank=0).mlir_tensor_type_asm
      'tensor<f32>'
      >>> ArrayParams(dtype=np.float32, shape=()).mlir_tensor_type_asm
      'tensor<f32>'

    Shaped:
      >>> ArrayParams(dtype=np.float32, shape=(2, 3)).mlir_tensor_type_asm
      'tensor<2x3xf32>'
      >>> ArrayParams(dtype=np.float32, shape=(-1, 3)).mlir_tensor_type_asm
      'tensor<?x3xf32>'
    """
    if self.dtype is Unspec:
      dtype_asm = "!numpy.any_dtype"
    else:
      dtype_asm = _dtype_to_mlir_asm(self.dtype)
      if not dtype_asm:
        raise ValueError("Unsupported MLIR tensor element type %r" %
                         (self.dtype,))
    if self.shape is Unspec:
      shape_asm = "*"
    else:
      shape_asm = "x".join((str(d) if d >= 0 else "?") for d in self.shape)
    if shape_asm:
      shape_asm += "x"
    return "tensor<%s%s>" % (shape_asm, dtype_asm)

  def new_ndarray(self):
    """Creates a new ndarray from these params.

      >>> ArrayParams().new_ndarray()
      Traceback (most recent call last):
      ...
      ValueError: ArrayParams(dtype=Unspec) is not concrete
      >>> ArrayParams(np.float32, (1, 2)).new_ndarray() * 0.0
      array([[0., 0.]], dtype=float32)
    """
    if not self.is_concrete:
      raise ValueError("%r is not concrete" % (self,))
    return np.ndarray(dtype=self.dtype, shape=self.shape)


class TypeConstraint:
  """Base class for type constraints."""
  pass


class TypeConstraints(list):
  """Collection of type constraints.

    >>> TypeConstraints([DynamicDim()])
    TypeConstraints(DimFlag(Dynamic, Unspec))
    >>> TypeConstraints([DynamicDim(), Rank(4)])
    TypeConstraints(DimFlag(Dynamic, Unspec), Rank(4))
    >>> TypeConstraints(DynamicDim(), Rank(4))
    TypeConstraints(DimFlag(Dynamic, Unspec), Rank(4))
    >>> TypeConstraints(Rank(4))
    TypeConstraints(Rank(4))
    >>> TypeConstraints("foobar")
    Traceback (most recent call last):
    ...
    AssertionError
  """

  def __init__(self, *constraints):
    if len(constraints) == 1 and not isinstance(constraints[0],
                                                ArrayConstraint):
      constraints = constraints[0]
    super().__init__(constraints)
    assert (all(isinstance(c, ArrayConstraint) for c in self))

  def __repr__(self):
    return "TypeConstraints(%s)" % (", ".join([repr(c) for c in self]))

  def all_of(self, clazz):
    """Finds all of the given class."""
    return [c for c in self if isinstance(c, clazz)]

  def one_of(self, clazz):
    """Finds at most one constraint of the given class."""
    found = [c for c in self if isinstance(c, clazz)]
    if not found:
      return None
    if len(found) > 1:
      raise ValueError("Conflicting constraints. Expected one of %r. Got %r" %
                       (clazz, found))
    return found[0]


class ArrayConstraint(TypeConstraint):
  """Base class for a constraint on an array's characteristics."""

  def implies_dtype(self):
    return False

  @property
  def dtype(self):
    raise NotImplementedError()

  def implies_rank(self):
    return False

  @property
  def rank(self):
    raise NotImplementedError()

  def implies_dims(self):
    return False

  @property
  def dims(self):
    raise NotImplementedError()

  def implies_dim_flag(self):
    return False

  @property
  def dim_flag(self):
    raise NotImplementedError()


class DType(ArrayConstraint):
  """A constraint on a dtype.

  DType constraints are exclusive with only one permitted in a set.

    >>> DType(np.float32)
    DType(float32)
    >>> DType("foobar")
    Traceback (most recent call last):
    ...
    AssertionError
  """
  __slots__ = ["_dtype"]

  def __init__(self, dtype):
    super().__init__()
    assert isinstance(dtype, type)
    self._dtype = dtype

  @property
  def dtype(self):
    return self._dtype

  def implies_dtype(self):
    return True

  def __repr__(self):
    return "DType(%s)" % (self._dtype.__name__,)


class Rank(ArrayConstraint):
  """Establishes a fixed rank for the array.

    >>> Rank(1)
    Rank(1)
    >>> Rank(0)
    Rank(0)
    >>> Rank(-1)
    Traceback (most recent call last):
    ...
    AssertionError
    >>> Rank("foobar")
    Traceback (most recent call last):
    ...
    AssertionError

  """
  __slots__ = ["_rank"]

  def __init__(self, rank):
    super().__init__()
    assert (isinstance(rank, int) and rank >= 0)
    self._rank = rank

  @property
  def rank(self):
    return self._rank

  def implies_rank(self):
    return True

  def __repr__(self):
    return "Rank(%d)" % (self._rank)


class Shape(ArrayConstraint):
  """Establishes a static shape for an array.

  All dimensions must be a non-negative integer or Unspec.

    >>> Shape(1, 2, 3)
    Shape(1, 2, 3)
    >>> Shape(Unspec, 1)
    Shape(Unspec, 1)
    >>> Shape()
    Shape()
    >>> Shape(-1, 1)
    Traceback (most recent call last):
    ...
    AssertionError
  """
  __slots__ = ["_dims"]

  def __init__(self, *dims):
    super().__init__()
    assert (all(d is Unspec or (isinstance(d, int) and d >= 0) for d in dims))
    self._dims = tuple(dims)

  @property
  def dims(self):
    return self._dims

  def implies_dims(self):
    return True

  @property
  def rank(self):
    return len(self._dims)

  def implies_rank(self):
    return True

  def __repr__(self):
    return "Shape(%s)" % (", ".join(str(d) for d in self._dims))


class DimFlagEnum(_LiterateEnum):
  """Flag for the kind of DimFlag constraint."""
  Dynamic = 1


class DimFlag(ArrayConstraint):
  """Generic flag applying to one or more dimensions.

  If dims is Unspec, the flag applies to all dims.

    >>> DimFlag("Dynamic")
    DimFlag(Dynamic, Unspec)
    >>> DimFlag("Dynamic", 1)
    DimFlag(Dynamic, (1,))
    >>> DimFlag("Dynamic", (0, 1))
    DimFlag(Dynamic, (0, 1))
  """
  __slots__ = ["_flag", "_dims"]

  def __init__(self, flag, dims=Unspec):
    super().__init__()
    self._flag = DimFlagEnum.parse(flag)
    if isinstance(dims, int):
      assert (dims >= 0)
      self._dims = (dims,)
    elif dims is Unspec:
      self._dims = Unspec
    else:
      self._dims = tuple(dims)
      assert (all(isinstance(d, int) and d >= 0 for d in self._dims))

  def implies_dim_flag(self):
    return False

  @property
  def dim_flag(self):
    return self._flag, self._dims

  def __repr__(self):
    return "DimFlag(%r, %r)" % (self._flag, self._dims)


def DynamicDim(dims=Unspec):
  """Dim flag that signals a dimension should be considered dynamic."""
  return DimFlag(DimFlagEnum.Dynamic, dims)


if __name__ == "__main__":
  import doctest
  doctest.testmod()
