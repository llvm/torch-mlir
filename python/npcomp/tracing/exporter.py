#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import inspect
import numpy as np
from typing import Optional

from ..meta.types import *

__all__ = [
    "Exporter",
    "ExportFunction",
    "ExportPyFunction",
]


def _value_type_from_annotation(annotation):
  # TODO: This is just enough to recognize ndarrays.
  if annotation is np.ndarray:
    return ValueType(TypeClass.NdArray)
  else:
    return ValueType()


def _signature_from_pyfunc(pyfunc):
  pysig = inspect.signature(pyfunc)
  sig = Signature(len(pysig.parameters))
  # Arguments
  for i, param in enumerate(pysig.parameters.values()):
    if param.kind not in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
      raise ValueError(
          "Currently only positional function signature are supported")

    sig.arg_names[i] = param.name
    annot = param.annotation
    if annot is param.empty:
      continue
    sig.args[i] = _value_type_from_annotation(annot)

  # Result
  if pysig.return_annotation is not pysig.empty:
    sig.result = _value_type_from_annotation(pysig.return_annotation)

  return sig


class ExportFunction:
  """Base class for functions that can be exported."""
  __slots__ = ["_sig"]

  def __init__(self, sig=None):
    self._sig = sig if sig else Signature()

  @property
  def sig(self):
    return self._sig

  def __repr__(self):
    return "def %r" % self._sig


class ExportPyFunction(ExportFunction):
  """Wraps a fully specialized python function that is staged for export.

  At different phases of compilation, the wrapped function will be
  treated differently. At the initial phase, it is just a pass-through
  and provides introspection capabilities.

  Basic access:
    >>> def simple(a, b): return a + b
    >>> ExportPyFunction(simple)
    pydef simple(a: Any, b: Any) -> Any
    >>> def mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ...   return a * b
    >>> ExportPyFunction(mul)
    pydef mul(a: NdArray, b: NdArray) -> NdArray
    >>> ExportPyFunction(mul).sig
    (a: NdArray, b: NdArray) -> NdArray

  Manipulating the signature:
    >>> f = ExportPyFunction(mul)
    >>> f.sig.args["a"] += Rank(2)
    >>> f.sig.args["b"] = "Any"
    >>> f.sig.result += Shape(1, 2)
    >>> f
    pydef mul(a: NdArray[Rank(2)], b: Any) -> NdArray[Shape(1, 2)]
  """
  __slots__ = ExportFunction.__slots__ + ["_pyfunc", "__name__"]

  def __init__(self, pyfunc, name=None):
    super().__init__(sig=_signature_from_pyfunc(pyfunc))
    assert (hasattr(pyfunc, "__call__") and
            hasattr(pyfunc, "__name__")), "Not a python function"
    self._pyfunc = pyfunc
    self.__name__ = name if name else pyfunc.__name__

  @property
  def pyfunc(self):
    return self._pyfunc

  def __repr__(self):
    return "pydef %s%r" % (self.__name__, self._sig)

  def __call__(self, *args, **kwargs):
    return self._pyfunc(*args, **kwargs)


class _ExpandoNode:
  """Expando object that can be indexed into to construct a namespace."""
  __slots__ = [
      "_parent", "_services", "_local_name", "_parent_name", "_children",
      "_attached"
  ]

  def __init__(self, parent: Optional["_ExpandoNode"], services: "_Services",
               local_name: str):
    super().__init__()
    object.__setattr__(self, "_parent", parent)
    object.__setattr__(self, "_services", services)
    object.__setattr__(self, "_local_name", local_name)
    object.__setattr__(self, "_parent_name",
                       parent._get_full_name() if parent else "")
    object.__setattr__(self, "_children", {})
    object.__setattr__(self, "_attached", parent is None)

  def _attach(self):
    if self._attached:
      return
    if self._local_name in self._parent._children:
      raise KeyError("Cannot re-assign '%s'" % (self._get_full_name(),))
    self._parent._attach()
    self._parent._children[self._local_name] = self
    object.__setattr__(self, "_attached", True)

  def _get_full_name(self):
    if not self._parent:
      return ""  # Root is always empty name.
    full_name = (self._parent_name + "." +
                 self._local_name if self._parent_name else self._local_name)
    return full_name

  def _get_child_name(self, child_local_name):
    full_name = self._get_full_name()
    if not full_name:
      return child_local_name
    else:
      return full_name + "." + child_local_name

  def __repr__(self):
    return "Namespace(\"%s\")" % (self._get_full_name())

  def __contains__(self, key):
    return key in self._children

  def __getitem__(self, key):
    key = str(key)
    existing = self._children.get(key)
    if existing is not None:
      return existing
    # Speculatively create a child expando.
    child = _ExpandoNode(self, self._services, key)
    return child

  def __setitem__(self, key, value):
    if not inspect.isfunction(value):
      raise TypeError("Cannot assign value to an exporter: %r" % (value,))
    child_name = self._get_child_name(key)
    if key in self._children:
      # TODO: Relax this once __delitem__ is implemented.
      raise KeyError("Cannot re-assign '%s'" % (child_name))
    self._attach()
    self._children[key] = self._services.wrap_function(value, child_name)

  def __getattr__(self, name):
    return self[name]

  def __setattr__(self, name, value):
    try:
      self[name] = value
    except KeyError as e:
      raise AttributeError(str(e)) from None

  def __dir__(self):
    return self._children.keys()


class _Services:
  """Services and support for the Exporter.

  Exporters are user objects, so most of the functional components are
  contained in the associated _Services object.
  """

  def wrap_function(self, f, full_name):
    if isinstance(f, ExportFunction):
      return f
    # TODO: Need to scan through providers and choose.
    return ExportPyFunction(f, name=full_name)


class Exporter:
  """Top-level UI object for assembling a program for export.

  The exporter defines an open namespace of functions to be exported.
  Logically, it can be thought of as a dict-of-dicts that is populated
  by assignment of functions to leaves. The act of assigning a function
  captures it as an ExportFunction and binds it to the exporter. This
  ExportFunction exposes the object model that can be manipulated to
  refine the compiled form. By default, any calls to such functions will
  delegate to the original function, capturing examples that constrain
  and allow further optimizations on the compiled form.

  There are several reserved names that can not have functions bound
  to them with the dot notation, but can still be referenced by subscripting
  if necessary:
    TODO: Reserved names. 'captures', etc.

    >>> exp = Exporter()
    >>> exp
    Exporter()

  Creating namespaces and functions with attribute access:
    >>> exp = Exporter()
    >>> exp.ns1
    Namespace("ns1")
    >>> "ns1" in exp  # Not yet attached
    False
    >>> exp.ns1.ns2.f = lambda x: x
    >>> exp.ns1.ns2  # Should be attached
    Namespace("ns1.ns2")
    >>> exp.ns1.ns2.f
    pydef ns1.ns2.f(x: Any) -> Any

  Via index access:
    >>> exp = Exporter()
    >>> exp["ns1"]["f"] = lambda x: x
    >>> dir(exp["ns1"])
    ['f']
    >>> exp["ns1"]["f"]
    pydef ns1.f(x: Any) -> Any

  Illegal access:
    >>> exp = Exporter()
    >>> exp.ns1.ns2.f = lambda x: x
    >>> exp.ns1.ns2.f = lambda x: x
    Traceback (most recent call last):
    ...
    AttributeError: "Cannot re-assign 'ns1.ns2.f'"
    >>> exp.ns1 = lambda x: x
    Traceback (most recent call last):
    ...
    AttributeError: "Cannot re-assign 'ns1'"
  """
  __slots__ = ["_root", "_services"]

  def __init__(self):
    super().__init__()
    services = _Services()
    object.__setattr__(self, "_root", _ExpandoNode(None, services, ""))
    object.__setattr__(self, "_services", services)

  def __repr__(self):
    return "Exporter()"

  def __contains__(self, key):
    return key in self._root

  def __getitem__(self, key):
    return self._root[key]

  def __setitem__(self, key, value):
    self._root[key] = value

  def __getattr__(self, name):
    return getattr(self._root, name)

  def __setattr__(self, name, value):
    setattr(self._root, name, value)


if __name__ == "__main__":
  import doctest
  doctest.testmod()
