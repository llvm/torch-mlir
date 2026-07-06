# Part of the LLVM Project, under the Apache License v2.0 WITH LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

from typing import Any

import torch
import torch.fx as fx

_AS_STRIDED = torch.ops.aten.as_strided.default
_INDEX = torch.ops.aten.index.Tensor
_LIST = (list, tuple, fx.immutable_collections.immutable_list)
_MISSING = object()


def rewrite_as_strided(g: fx.Graph) -> bool:
    """Rewrite supported FX ``aten.as_strided.default`` nodes in place.

    Torch IR does not carry storage identity, physical strides, or storage
    offsets, so this pass runs while FakeTensor metadata is still available. It
    replaces each supported ``as_strided`` read with ``aten.index.Tensor`` on a
    base tensor that still has the needed storage layout. The rewrite preserves
    result values for supported static cases; it does not preserve view aliasing.

    The graph must have an owning ``GraphModule`` because generated index
    tensors are attached as synthetic buffers and read back through ``get_attr``
    nodes. Dynamic requests, out-of-bounds requests, and requests whose storage
    offsets cannot be mapped to base indices raise ``ValueError`` before Torch
    IR import. Returns ``True`` if the graph changed.
    """
    changed = False
    # Walk a snapshot from the back so nested as_strided users are rewritten
    # while their as_strided producers are still available for base tracing.
    # Because this walks a snapshot, it can also rewrite producers that become
    # dead. Graph DCE removes dead replacement/get_attr nodes. Their synthetic
    # buffers may remain on the GraphModule, but import only reads live get_attr
    # references.
    for node in reversed(list(g.nodes)):
        if node.op != "call_function" or node.target != _AS_STRIDED:
            continue
        replacement = _rewrite(g, node)
        node.replace_all_uses_with(replacement)
        g.erase_node(node)
        changed = True
    if changed:
        g.eliminate_dead_code()
        g.lint()
    return changed


def _rewrite(g: fx.Graph, node: fx.Node) -> fx.Node:
    # as_strided addresses the source tensor's storage, not its logical value.
    # If storage_offset is omitted, PyTorch uses source.storage_offset(). If it
    # is supplied, it is already absolute in that storage, not relative to
    # source[0]. _indices subtracts the selected base tensor's storage_offset()
    # before expressing those addresses as logical indices into the base.
    source = _arg(node, 0, "self")
    size = _ints(_arg(node, 1, "size"), "size")
    stride = _ints(_arg(node, 2, "stride"), "stride")
    offset = _arg(node, 3, "storage_offset", None)
    source_value = _tensor(source, "self")
    offset = (
        _int(source_value.storage_offset(), "self.storage_offset")
        if offset is None
        else _int(offset, "storage_offset")
    )

    base = _base(source)
    base_value = _tensor(base, "base")
    indices = _indices(
        _ints(base_value.shape, "base.shape", meta=True),
        _ints(base_value.stride(), "base.stride", meta=True),
        _int(base_value.storage_offset(), "base.storage_offset"),
        size,
        stride,
        offset,
    )
    index_nodes = [_constant(g, node, i, v) for i, v in enumerate(indices)]
    with g.inserting_before(node):
        replacement = g.call_function(_INDEX, args=(base, index_nodes))
    replacement.meta.update(node.meta)
    replacement.meta["val"] = torch.empty(
        tuple(size), dtype=_tensor(node, "result").dtype
    )
    return replacement


def _constant(g: fx.Graph, node: fx.Node, index: int, tensor: torch.Tensor) -> fx.Node:
    # aten.index needs tensor index operands. Attach each generated tensor as a
    # synthetic GraphModule buffer and reference it with get_attr so the existing
    # literal-import path can read it. These buffers are import-time literals,
    # not user-authored model state.
    gm = node.graph.owning_module
    if gm is None:
        raise ValueError("aten.as_strided.default rewrite needs an owning GraphModule")
    name = f"_torch_mlir_as_strided_{node.name}_index_{index}"
    while hasattr(gm, name):
        name += "_"
    gm.register_buffer(name, tensor)
    with g.inserting_before(node):
        const_node = g.get_attr(name)
    const_node.meta["val"] = tensor
    return const_node


def _arg(node: fx.Node, index: int, name: str, default: Any = _MISSING) -> Any:
    # FX may use positional or keyword schema arguments depending on exporter
    # normalization. Keep this local so the rest of the pass reads schema names.
    if index < len(node.args):
        return node.args[index]
    if name in node.kwargs or default is not _MISSING:
        return node.kwargs.get(name, default)
    raise ValueError(f"aten.as_strided.default missing required argument `{name}`")


def _ints(values: Any, name: str, meta: bool = False) -> tuple[int, ...]:
    # torch.export/ATen already rejects schema-invalid as_strided calls such as
    # rank-mismatched size/stride lists, negative result sizes, and negative
    # strides. The extra importer check is that these values are concrete ints
    # before Torch IR loses storage metadata.
    if isinstance(values, _LIST) and all(type(value) is int for value in values):
        return tuple(values)
    expected = "static before Torch IR import" if meta else "a static int list"
    raise ValueError(f"aten.as_strided.default `{name}` must be {expected}")


def _int(value: Any, name: str) -> int:
    if type(value) is int:
        return value
    raise ValueError(
        f"aten.as_strided.default `{name}` must be static before Torch IR import"
    )


def _tensor(value: Any, name: str) -> torch.Tensor:
    # This runs after torch.export fake tensor propagation. Every FX tensor node
    # inspected here must have meta["val"], otherwise the graph does not carry
    # the storage metadata needed for a safe rewrite.
    value = value.meta.get("val") if isinstance(value, fx.Node) else value
    if isinstance(value, torch.Tensor):
        return value
    raise ValueError(f"aten.as_strided.default `{name}` is missing FakeTensor metadata")


def _base(value: Any) -> Any:
    # Here the base is the FX tensor that replacement aten.index.Tensor will
    # index. Follow FakeTensor storage identity backward through view-like ops to
    # the nearest graph storage boundary. This is not necessarily PyTorch
    # Tensor._base or the original allocation owner. Copy or layout-conversion
    # results are the new base because they have new storage.
    while isinstance(value, fx.Node):
        operand = _storage_operand(value)
        if operand is None:
            return value
        value = operand
    return value


def _storage_operand(node: fx.Node) -> fx.Node | None:
    # Return the tensor operand that shares storage with node. View-like nodes
    # have one such operand. Copy and layout-changing nodes have none and stop
    # the walk.
    value = _tensor(node, "result")
    for arg in list(node.args) + list(node.kwargs.values()):
        match = _same_storage_arg(value, arg)
        if match is not None:
            return match
    return None


def _same_storage_arg(value: torch.Tensor, arg: Any) -> fx.Node | None:
    # FX args can be scalar nodes or nested containers. Recurse only through
    # containers, so scalar metadata nodes are never treated as storage
    # ancestors.
    if isinstance(arg, fx.Node):
        arg_value = arg.meta.get("val")
        if isinstance(arg_value, torch.Tensor):
            if value.untyped_storage() is arg_value.untyped_storage():
                return arg
    if isinstance(arg, _LIST):
        for item in arg:
            match = _same_storage_arg(value, item)
            if match is not None:
                return match
    return None


def _indices(
    base_shape: tuple[int, ...],
    base_stride: tuple[int, ...],
    base_offset: int,
    size: tuple[int, ...],
    stride: tuple[int, ...],
    offset: int,
) -> tuple[torch.Tensor, ...]:
    """Build per-dimension index tensors for the selected FX base tensor.

    ``offset`` is absolute in the source storage. ``base_offset`` converts those
    addresses to offsets relative to the selected base. The decoder supports
    dense row-major layouts, dense channels-last-like layouts, and views that
    trace back to one of those bases. It is not an exhaustive solver for
    arbitrary or overlapping stride systems. Accepted requests must decode with
    no residue and stay within ``base_shape``.
    """
    # Compute each result element's storage offset relative to the base:
    #   offset - base_offset + sum(result_index[d] * stride[d])
    offsets = torch.full(tuple(size), offset - base_offset, dtype=torch.int64)
    for dim, step in enumerate(stride):
        if step:
            shape = [1] * len(size)
            shape[dim] = size[dim]
            offsets += torch.arange(size[dim], dtype=torch.int64).reshape(shape) * step

    # Decode high-stride dimensions first. Zero-stride base dimensions keep
    # coordinate zero because they do not contribute to storage offsets.
    coords: list[torch.Tensor] = [torch.zeros_like(offsets)] * len(base_shape)
    remaining = offsets
    dims = sorted(range(len(base_shape)), key=lambda i: base_stride[i], reverse=True)
    for dim in dims:
        if base_stride[dim]:
            coord = torch.div(remaining, base_stride[dim], rounding_mode="floor")
            coords[dim] = coord
            remaining -= coord * base_stride[dim]

    # A nonzero residue means the greedy decoder could not express at least one
    # requested storage offset as integer base coordinates.
    if _any(remaining != 0):
        raise ValueError(
            "aten.as_strided.default storage offsets cannot be mapped to base indices"
        )

    # Zero residue only means the decoder found integer coordinates. Bounds
    # still reject coordinates outside the selected base tensor.
    for dim, coord in enumerate(coords):
        if _any(coord < 0) or _any(coord >= base_shape[dim]):
            raise ValueError("aten.as_strided.default indexes outside the base tensor")
    return tuple(coords)


def _any(value: torch.Tensor) -> bool:
    # Empty predicate tensors come from empty results; treat them as false before
    # converting torch.any's scalar result to bool.
    return value.numel() != 0 and bool(torch.any(value))
