#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Base classes and interfaces for mapping ops to MLIR.

The goal of this facility is to define the majority of op mappings by example,
from the actual Python invocation, using the PyTorch tracer to extract its
Graph IR, and introspecting that to determine mappings, metadata and types
for corresponding MLIR definitions. This is not meant to cover everything,
and it is expected that a number of hard ops should be mapped by hand.

The result of building such an OpRegistry should be a data structure that
can be used to generate ODS and tables for doing systematic imports from
a PyTorch Graph to corresponding MLIR module.

Example usage (fully automatic discovery):

  >>> r = OpRegistry()
  >>> r.op(torch.add,
       TensorValue("input"),
       TensorValue("other"),
       alpha=ScalarValue()).with_outref_variant()
"""

from typing import Dict, List, Optional, Sequence, Tuple

import logging
import random
import torch

__all__ = [
    "SimpleOpMapping",
    "OpRegistry",
    "ScalarValue",
    "TensorOutRef",
    "TensorValue",
]


def _is_same_value(value1, value2):
  # Tensors are considered via reference equality.
  value1_tensor = isinstance(value1, torch.Tensor)
  value2_tensor = isinstance(value2, torch.Tensor)
  if value1_tensor or value2_tensor:
    return value1 is value2

  # Everything else is value equality.
  return value1 == value2


def _extract_immediate(node):
  """Extracts an immediate value from a node.

  Supported node types:
    prim::Constant
    prim::ListConstruct
  """
  # Constant
  if node.kind() == "prim::Constant":
    # Try to extract as different types.
    try:
      return node.t("value")
    except RuntimeError:
      pass
    try:
      return node.f("value")
    except RuntimeError:
      pass
    try:
      return node.i("value")
    except RuntimeError:
      pass
    try:
      return node.s("value")
    except RuntimeError:
      pass
    return None
  # List
  elif node.kind() == "prim::ListConstruct":
    return [_extract_immediate(i.node()) for i in node.inputs()]
  else:
    raise ValueError("Unrecognized immediate input node: {!r}".format(node))


class ValueSpec:
  """Base class for inputs to operations.

  This binds information about how the input is mapped to the MLIR operation.
  """

  def __init__(self, name=None):
    super().__init__()
    self.name = name

  @property
  def mlir_ods_predicate(self):
    return "AnyType"

  def generate_example(self, index=0):
    """Generates an example value."""
    raise NotImplementedError()

  def __repr__(self):
    return "{}({!r})".format(self.__class__.__name__, self.name)


class TensorValue(ValueSpec):
  """An input that is a tensor."""

  def __init__(self, name=None, *, example_size=None):
    super().__init__(name=name)
    if example_size is None:
      example_size = (2, 3, 7)  # No significance.
    self.example_size = example_size

  @property
  def mlir_ods_predicate(self):
    return "ATenAnyTensor"

  def generate_example(self, index=0):
    return torch.rand(*self.example_size)


class TensorOutRef(ValueSpec):
  """A tensor that is passed by ref as an out parameter."""

  def __init__(self, name=None, *, example_size=None):
    super().__init__(name=name)
    if example_size is None:
      example_size = (2, 3, 7)  # No significance.
    self.example_size = example_size

  @property
  def mlir_ods_predicate(self):
    return "ATenAnyRefTensor"

  def generate_example(self, index=0):
    return torch.rand(*self.example_size)


class ScalarValue(ValueSpec):
  """An input that is a scalar."""

  def __init__(self, name=None, value=None):
    super().__init__(name=name)
    self.value = value

  @property
  def mlir_ods_predicate(self):
    return "ATenAnyScalar"

  def generate_example(self, index=0):
    if self.value is not None:
      return self.value
    return 1.0 + index  # Generates a stable value.


class OpMapping:
  """Base class for things purporting to map an operation."""
  pass


class SimpleOpMapping(OpMapping):
  """Maps a PyTorch invocation to its MLIR representation."""

  def __init__(self, op_f, *op_args, **op_kwargs):
    super().__init__()
    self.op_f = op_f
    self.op_args = op_args
    self.op_kwargs = op_kwargs
    self.outref_variant_value = None  # type: Optional[TensorOutRef]

    # Set after finalize.
    self.op_kind = None  # type: Optional[str]
    self.op_arity = -1  # type: int
    self.operand_map = None  # type: Optional[List[Tuple[int, ValueSpec]]]
    self.result_map = None  # type: Optional[List[Tuple[int, ValueSpec]]]
    self.mlir_operation_name = None  # type: Optional[str]

  def __repr__(self):
    return ("SimpleOp({kind!r}[{arity}] -> {name!s}, operands={operands!r}, "
            "results={results!r})".format(kind=self.op_kind,
                                          arity=self.op_arity,
                                          name=self.mlir_operation_name,
                                          operands=self.operand_map,
                                          results=self.result_map))

  def clone(self) -> "SimpleOpMapping":
    copy = SimpleOpMapping(self.op_f, *self.op_args, **self.op_kwargs)
    for name in [
        "outref_variant_value", "op_kind", "op_arity", "operand_map",
        "result_map", "mlir_operation_name"
    ]:
      setattr(copy, name, getattr(self, name))
    return copy

  def with_outref_variant(self, value=None):
    """Instructs the registry to also generate an outref variant.

    This is done by cloning the op prior to finalizing and adding an out=
    paramer.
    """
    self.outref_variant_value = TensorOutRef() if value is None else value
    return self

  @property
  def all_arg_values(self) -> List[ValueSpec]:
    """Returns all arg values (either positional or kw)."""
    return list(self.op_args) + list(self.op_kwargs.values())

  @property
  def is_outref_form(self) -> bool:
    """Whether the op contains an out parameter that aliases to the result."""
    return any(isinstance(a, TensorOutRef) for a in self.all_arg_values)

  def generate_example(self) -> Tuple[Tuple, Dict]:
    """Generates an example signature for invoking the op.

    Returns:
      (tuple, dict) of positional and keyword args.
    """
    index = 0
    positional = list()
    kw = dict()
    for op_arg in self.op_args:
      positional.append(op_arg.generate_example(index))
      index += 1
    for kw_name, kw_value in self.op_kwargs.items():
      kw[kw_name] = kw_value.generate_example(index)
      index += 1
    return positional, kw

  def finalize(self):
    """Finalizes the mapping once all hints have been applied."""
    # Update the name on all args if undefined.
    for index, op_arg in enumerate(self.op_args):
      if op_arg.name is None:
        op_arg.name = "arg%d".format(index)
    for key, op_arg in self.op_kwargs.items():
      if op_arg.name is None:
        op_arg.name = key

    # Create an example graph and configure from it.
    self._configure_from_example()

    # Determine default operation name.
    if self.mlir_operation_name is None:
      self._set_default_mlir_operation_name()

  def _set_default_mlir_operation_name(self):
    op_ns, op_name = self.op_kind.split("::", maxsplit=1)
    # Since these are emitted into the "aten" dialect namespace, alias them
    # to a prefix of "builtin" to distinguish from custom ops and others.
    if op_ns == "aten":
      op_ns = "builtin"
    default_name = op_ns + "." + op_name

    if self.is_outref_form:
      default_name += "_outref"
    self.mlir_operation_name = default_name

  def _configure_from_example(self):
    # Trace the op so that we get an example graph like this:
    #   %0 : Float(2, 3, 7) = prim::Constant[value=<Tensor>]()
    #   %1 : Float(2, 3, 7) = prim::Constant[value=<Tensor>]()
    #   %2 : float = prim::Constant[value=3.]()
    #   %3 : Float(2, 3, 7) = aten::add(%0, %1, %2)
    #   return (%3)
    # The def of the return value is expected to be the modeled op. The
    # inputs to that op are expected to be captured constants that can be
    # re-associated to the example inputs.
    example_args, example_kwargs = self.generate_example()

    def forward():
      return self.op_f(*example_args, **example_kwargs)

    trace = torch.jit.trace(forward, tuple())
    graph = trace.graph
    logging.debug("Graph for op %r: %s", self.op_f, graph)

    # Track up from the return node and assume this is our op.
    return_node = graph.return_node()
    return_inputs = list(return_node.inputs())
    assert len(return_inputs) == 1, "Expected one input return"
    op_node = return_inputs[0].node()
    op_inputs = list(op_node.inputs())
    logging.debug("Found op node: %r", op_node)

    # Meta-data about the source op.
    self.op_kind = op_node.kind()
    self.op_arity = len(op_inputs)
    if self.operand_map is None:
      self.operand_map = self._associate_inputs(op_inputs, example_args,
                                                example_kwargs)

    # Results.
    op_outputs = list(op_node.outputs())
    if self.result_map is None:
      if self.is_outref_form:
        # Only support single outref results.
        assert len(op_outputs) == 1, (
            "For outref ops, only a single output is supported")
        self.result_map = [(0, TensorOutRef("result"))]
      else:
        # Map results in order.
        self.result_map = []

        def result_name(i):
          if len(op_outputs) == 1:
            return "result"
          else:
            return "result%d" % i

        for i, op_output in enumerate(op_outputs):
          op_output_type = op_output.type()
          if issubclass(type(op_output_type), torch.TensorType):
            self.result_map.append((i, TensorValue(result_name(i))))
          else:
            raise ValueError(
                "Unsupported op output type: {!r}".format(op_output_type))
    return self

  def _associate_inputs(self, op_inputs, example_args, example_kwargs):
    """Given inputs to a graph op node, associates to the input args.

    This will match up example arguments with what was produced in the graph,
    setting the operand_map.

    Returns:
      List of (input_index, ValueSpec) mapping inputs to the graph node to
      provided values in the op definition.
    """
    assert len(example_args) == len(self.op_args)
    assert example_kwargs.keys() == self.op_kwargs.keys()

    def find_arg(value):
      for i, arg in enumerate(example_args):
        if _is_same_value(arg, value):
          return self.op_args[i]
      for key, arg in example_kwargs.items():
        if _is_same_value(arg, value):
          return self.op_kwargs[key]

      raise KeyError("Op input not in arguments: {!r} -> {!r}".format(
          value, op_inputs))

    operand_map = []
    for i, op_input in enumerate(op_inputs):
      input_node = op_input.node()
      immediate_value = _extract_immediate(input_node)
      if immediate_value is not None:
        operand_map.append((i, find_arg(immediate_value)))
    return operand_map


class OpRegistry:
  """Maintains a registry of op mappings."""

  def __init__(self):
    super().__init__()
    self._mappings = []
    self._pending_mapping = None

  def op(self, op_f, *op_args, **op_kwargs):
    """Forwards to the SimpleOpMapping constructor and adds it.

    The mapping is not finalized until either the registry is finalized or the
    next op mapping is added. This allows tweaks to the mapping to be done
    inline prior to performing detailed introspection.

    Returns:
      The SimpleOpMapping instance.
    """
    self._finalize_pending()
    m = SimpleOpMapping(op_f, *op_args, **op_kwargs)
    self._pending_mapping = m
    return m

  @property
  def mappings(self) -> Sequence[OpMapping]:
    """Returns the list of OpMapping.

    Returns:
      Sequence of OpMapping concrete classes (most commonly SimpleOpMapping).
    """
    self._finalize_pending()
    return self._mappings

  def _finalize_pending(self):
    if not self._pending_mapping:
      return

    outref_mapping = None
    pending_mapping = self._pending_mapping
    self._pending_mapping = None
    if pending_mapping.outref_variant_value:
      # Generate a variant (with an out= form).
      outref_mapping = pending_mapping.clone()
      outref_mapping.op_kwargs["out"] = outref_mapping.outref_variant_value
      outref_mapping.outref_variant_value = None

    # Finalize the original.
    pending_mapping.finalize()
    self._mappings.append(pending_mapping)

    # Finalize the outref form if generated.
    if outref_mapping:
      outref_mapping.finalize()
      self._mappings.append(outref_mapping)
