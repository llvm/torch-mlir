import torch
from torch._decomp import register_decomposition, global_decomposition_table
from torch._prims_common.wrappers import out_wrapper
from torch import Tensor
from torch._ops import OpOverload, OpOverloadPacket
from typing import Callable, Dict, Sequence, Union
from collections import defaultdict

# This updated registry and extra decomposition is used to deal with OPT-125M 
# that generates extra args when exporting through fx due to torch.tensor().
# Through this decomposition, we replace the need for that op. This is only
# used when model_name is specified as "opt-125M". Otherwise, normal flow.
registry_updated = global_decomposition_table["post_autograd"].copy()
registry_updated.pop(torch.ops.aten.maximum.default)


@register_decomposition(torch.ops.aten.maximum.default, registry=registry_updated)
@out_wrapper()
def maximum(
    self,
    other,
) -> Tensor:
    # If we are doing a maximum operation with a 0D tensor,
    # we should be doing a torch.clamp with a scalar instead anyways.
    if len(other.shape) == 0:
        return torch.clamp(self, min=other.item())
    else:
        # will never reach this else branch as we are only using this decomp
        # for opt-125M, but it would still work.
        return torch.ops.aten.maximum.out(self, other, out=None)

# default decompositions pulled from SHARK / torch._decomp
DEFAULT_DECOMPOSITIONS = [
    torch.ops.aten.embedding_dense_backward,
    torch.ops.aten.native_layer_norm_backward,
    torch.ops.aten.slice_backward,
    torch.ops.aten.select_backward,
    torch.ops.aten.norm.ScalarOpt_dim,
    torch.ops.aten.native_group_norm,
    torch.ops.aten.upsample_bilinear2d.vec,
    torch.ops.aten.split.Tensor,
    torch.ops.aten.split_with_sizes,
    torch.ops.aten.native_layer_norm,
    torch.ops.aten.masked_fill.Tensor,
    torch.ops.aten.masked_fill.Scalar,
    torch.ops.aten.t,
    torch.ops.aten.addmm,
    # decompositions that aid us in handling nn.BatchNorm2d
    torch.ops.aten._native_batch_norm_legit_functional,
    torch.ops.aten._native_batch_norm_legit_no_training,
    torch.ops.aten._native_batch_norm_legit,
    torch.ops.aten._native_batch_norm_legit.no_stats,
    torch.ops.aten.squeeze.dims,
    # decompositions for miscellaneous ops that are not handled in torch-mlir but have available decompositions
    torch.ops.aten.soft_margin_loss,
    torch.ops.aten.im2col,
    torch.ops.aten._euclidean_dist,
    torch.ops.aten.index_copy,
    torch.ops.aten.index_copy_,
    torch.ops.aten.grid_sampler_2d,
    torch.ops.aten.log_sigmoid_forward,
    torch.ops.aten.unsafe_split.Tensor,
    torch.ops.aten.binary_cross_entropy,
    torch.ops.aten.dot,
    torch.ops.aten._adaptive_avg_pool2d,
    torch.ops.aten._prelu_kernel,
    torch.ops.aten.full,
    torch.ops.aten._log_softmax,
    torch.ops.aten.nll_loss_forward,
    torch.ops.aten.nll_loss_backward,
    torch.ops.aten._to_copy,
    torch.ops.aten._log_softmax_backward_data,
    torch.ops.aten.lift_fresh_copy.default,
    torch.ops.aten._unsafe_index.Tensor,
    #manual decomposition for opt-125M (only included when user specifies model in API)
    torch.ops.aten.maximum,
]

def get_decompositions(
    aten_ops: Sequence[Union[torch._ops.OperatorBase, OpOverloadPacket]],
    registry,
    type: str = "post_autograd",
) -> Dict[torch._ops.OperatorBase, Callable]:
    assert type in {"post_autograd", "pre_autograd", "meta"}

    registry = registry_updated
    packets_to_overloads = defaultdict(list)
    for opo in registry:
        if isinstance(opo, (OpOverload, OpOverloadPacket)):
            packets_to_overloads[opo.overloadpacket].append(opo)
    decompositions: Dict[torch._ops.OperatorBase, Callable] = {}
    for op in aten_ops:
        if isinstance(op, OpOverloadPacket) and op in packets_to_overloads:
            for op_overload in packets_to_overloads[op]:
                decompositions[op_overload] = registry[op_overload]
        elif isinstance(op, (torch._ops.OperatorBase)) and op in registry:
            decompositions[op] = registry[op]
    return decompositions

def get_decomposition_table(model_name):
    if (model_name == "opt-125M"):
        return get_decompositions(DEFAULT_DECOMPOSITIONS, registry=registry_updated)
    else:
        return get_decompositions(DEFAULT_DECOMPOSITIONS[:-1], registry=global_decomposition_table["post_autograd"])
    