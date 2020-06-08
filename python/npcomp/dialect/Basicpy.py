#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from _npcomp.dialect import BasicpyDialectHelper as _BaseDialectHelper
from _npcomp.mlir import ir

__all__ = [
    "DialectHelper",
]


class DialectHelper(_BaseDialectHelper):
  r"""Dialect helper for the Basicpy dialect.

    >>> c = ir.MLIRContext()
    >>> h = DialectHelper(c, ir.OpBuilder(c))
    
  Dialect Types:
    >>> h.basicpy_NoneType
    !basicpy.NoneType
    >>> h.basicpy_EllipsisType
    !basicpy.EllipsisType
    >>> h.basicpy_SlotObject_type(
    ...     "foobar", h.basicpy_NoneType, h.basicpy_NoneType)
    !basicpy.SlotObject<foobar, !basicpy.NoneType, !basicpy.NoneType>

  singleton op:
    >>> m = c.new_module()
    >>> h.builder.insert_block_start(m.first_block)
    >>> _ = h.basicpy_singleton_op(h.basicpy_NoneType)
    >>> m.to_asm().strip()
    'module {\n  %0 = basicpy.singleton : !basicpy.NoneType\n}'

  slot_object ops:
    >>> m = c.new_module()
    >>> h.builder.insert_block_start(m.first_block)
    >>> v0 = h.basicpy_singleton_op(h.basicpy_NoneType).result
    >>> slot_object = h.basicpy_slot_object_make_op("foobar", v0, v0).result
    >>> _ = h.basicpy_slot_object_get_op(slot_object, 0)
    >>> print(m.to_asm().strip())
    module {
      %0 = basicpy.singleton : !basicpy.NoneType
      %1 = basicpy.slot_object_make(%0, %0) -> !basicpy.SlotObject<foobar, !basicpy.NoneType, !basicpy.NoneType>
      %2 = basicpy.slot_object_get %1[0] : !basicpy.SlotObject<foobar, !basicpy.NoneType, !basicpy.NoneType>
    }

  """

  def basicpy_binary_expr_op(self, result_type, lhs, rhs, operation_name):
    c = self.context
    attrs = c.dictionary_attr({"operation": c.string_attr(operation_name)})
    return self.op("basicpy.binary_expr", [result_type], [lhs, rhs], attrs)

  def basicpy_bool_constant_op(self, value):
    c = self.context
    ival = 1 if value else 0
    attrs = c.dictionary_attr({"value": c.integer_attr(self.i1_type, ival)})
    return self.op("basicpy.bool_constant", [self.basicpy_BoolType], [], attrs)

  def basicpy_bytes_constant_op(self, value):
    c = self.context
    attrs = c.dictionary_attr({"value": c.string_attr(value)})
    return self.op("basicpy.bytes_constant", [self.basicpy_BytesType], [],
                   attrs)

  def basicpy_binary_compare_op(self, lhs, rhs, operation_name):
    c = self.context
    attrs = c.dictionary_attr({"operation": c.string_attr(operation_name)})
    return self.op("basicpy.binary_compare", [self.basicpy_BoolType],
                   [lhs, rhs], attrs)

  def basicpy_singleton_op(self, singleton_type):
    return self.op("basicpy.singleton", [singleton_type], [])

  def basicpy_slot_object_make_op(self, class_name, *slot_values):
    c = self.context
    class_name_attr = c.string_attr(class_name)
    object_type = self.basicpy_SlotObject_type(class_name,
                                               *[v.type for v in slot_values])
    attrs = c.dictionary_attr({"className": class_name_attr})
    return self.op("basicpy.slot_object_make", [object_type], slot_values,
                   attrs)

  def basicpy_str_constant_op(self, value):
    c = self.context
    attrs = c.dictionary_attr({"value": c.string_attr(value.encode("utf-8"))})
    return self.op("basicpy.str_constant", [self.basicpy_StrType], [], attrs)

  def basicpy_unknown_cast_op(self, result_type, operand):
    return self.op("basicpy.unknown_cast", [result_type], [operand])


if __name__ == "__main__":
  import doctest
  doctest.testmod()
