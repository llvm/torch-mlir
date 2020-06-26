#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Utilities for matching and massaging python values."""

from collections import namedtuple
import weakref

_NotMapped = object()

__all__ = [
    "PyValueMap",
]


class HashableReference(namedtuple("HashableReference", "ref_id,referrent")):

  @staticmethod
  def create(referrent):
    try:
      return HashableReference(id(referrent), weakref.ref(referrent))
    except TypeError:
      # Fallback to value equality.
      return HashableReference(0, referrent)

  def __eq__(self, other):
    try:
      return self.ref_id == other.ref_id and self.referrent == other.referrent
    except AttributeError:
      return False

  def __hash__(self):
    return self.ref_id


class PyValueMap:
  """Maps between predicates that match python values and bound values.

  Maps to specific references:
    >>> class Refable: pass
    >>> refable1 = Refable()
    >>> refable2 = Refable()
    >>> pv = PyValueMap()
    >>> pv.bind_reference("unrefable", 1)
    >>> pv.bind_reference(refable1, 2)
    >>> pv.bind_reference(refable2, 3)
    >>> pv.lookup("unrefable")
    1
    >>> pv.lookup("nothere")
    >>> pv.lookup(refable2)
    3
    >>> pv.lookup(refable1)
    2

  Lookup by type:
    >>> pv.bind_type(Refable, 4)
    >>> pv.bind_type(str, 5)
    >>> pv.lookup(refable1)
    2
    >>> pv.lookup(Refable())
    4
    >>> pv.lookup("nothere")
    5
    >>> pv.lookup(999)
    >>> pv.bind_type(int, 6)
    >>> pv.lookup(999)
    6

  Predicate:
    >>> pv.lookup(1.2)
    >>> pv.bind_predicate(lambda x: x == 1.2, 7)
    >>> pv.lookup(1.2)
    7
  """
  __slots__ = [
      "_reference_map",
      "_type_map",
      "_type_filters",
      "_fallback_filters",
      "_validator",
  ]

  def __init__(self, validator=lambda x: True):
    super().__init__()
    self._reference_map = dict()  # of: dict[HashableReference, Any]
    self._type_map = dict()  # of: dict[Type, Any|_NotMapped]
    self._type_filters = list()  # of: list[(Type, Any)]
    self._fallback_filters = list()  # of: list[(lambda v, Any)]
    self._validator = validator

  def bind_reference(self, match_value, binding):
    assert self._validator(binding), "Illegal binding"
    self._reference_map[HashableReference.create(match_value)] = binding

  def bind_type(self, match_type, binding):
    assert isinstance(match_type, type)
    assert self._validator(binding), "Illegal binding"
    self._type_filters.append((match_type, binding))
    self._type_map.clear()  # Clears cached bindings

  def bind_predicate(self, predicate, binding):
    assert self._validator(binding), "Illegal binding"
    self._fallback_filters.append((predicate, binding))

  def lookup(self, value):
    # Check for direct reference equality.
    ref = HashableReference.create(value)
    binding = self._reference_map.get(ref)
    if binding is not None:
      return binding

    # Check the cached exact type match.
    match_type = type(value)
    binding = self._type_map.get(match_type)
    if binding is not None:
      return None if binding is _NotMapped else binding

    # Lookup by type filter.
    for predicate_type, binding in self._type_filters:
      if issubclass(match_type, predicate_type):
        self._type_map[match_type] = binding
        return binding

    # Fallback filters.
    for predicate, binding in self._fallback_filters:
      if predicate(value):
        return binding

    return None
