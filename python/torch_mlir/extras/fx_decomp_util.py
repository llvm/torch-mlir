import torch
from torch._decomp import get_decompositions

# Decompositions not in core_aten_decompositions() that torch-mlir needs.
# Note: only the specific overload matters — core may have other overloads of
# the same op but not the one torch.export produces.
DEFAULT_DECOMPOSITIONS = [
    torch.ops.aten.native_group_norm,
    torch.ops.aten.upsample_bilinear2d.vec,
    torch.ops.aten.native_layer_norm,
    torch.ops.aten.addmm,
    # decompositions that aid us in handling nn.BatchNorm2d
    torch.ops.aten._native_batch_norm_legit_functional,
    torch.ops.aten._native_batch_norm_legit_no_training,
    torch.ops.aten._native_batch_norm_legit,
    torch.ops.aten._native_batch_norm_legit.no_stats,
    torch.ops.aten.squeeze.dims,
    torch.ops.aten._euclidean_dist,
    torch.ops.aten.grid_sampler_2d,
    torch.ops.aten._adaptive_avg_pool2d,
    torch.ops.aten.full,
    torch.ops.aten._log_softmax,
    torch.ops.aten._to_copy,
    torch.ops.aten.diag,
]
if hasattr(torch.ops.aten, "_scaled_dot_product_flash_attention_for_cpu"):
    DEFAULT_DECOMPOSITIONS.append(
        torch.ops.aten._scaled_dot_product_flash_attention_for_cpu
    )
if hasattr(torch.ops.aten, "scaled_dot_product_attention"):
    DEFAULT_DECOMPOSITIONS.append(torch.ops.aten.scaled_dot_product_attention)


def get_decomposition_table():
    return get_expanded_decomposition_table()


_EXPANDED_DECOMP_EXCLUDE = [
    # FFT ops decompose to _fft_r2c/_fft_c2r which torch-mlir lacks
    torch.ops.aten.stft,
    torch.ops.aten.istft,
    torch.ops.aten.fft_fft,
    torch.ops.aten.fft_fft2,
    torch.ops.aten.fft_fftn,
    torch.ops.aten.fft_ifft,
    torch.ops.aten.fft_ifft2,
    torch.ops.aten.fft_ifftn,
    torch.ops.aten.fft_rfft,
    torch.ops.aten.fft_rfft2,
    torch.ops.aten.fft_rfftn,
    torch.ops.aten.fft_irfft,
    torch.ops.aten.fft_irfft2,
    torch.ops.aten.fft_irfftn,
    torch.ops.aten.fft_hfft,
    torch.ops.aten.fft_hfft2,
    torch.ops.aten.fft_hfftn,
    torch.ops.aten.fft_ihfft,
    torch.ops.aten.fft_ihfft2,
    torch.ops.aten.fft_ihfftn,
    torch.ops.aten.fft_fftshift,
    torch.ops.aten.fft_ifftshift,
    # norm/linalg_norm decompositions regress on complex tensors
    torch.ops.aten.norm,
    torch.ops.aten.linalg_norm,
    torch.ops.aten.linalg_vector_norm,
    # linalg_slogdet decomposition regresses (exposes unsupported path)
    torch.ops.aten.linalg_slogdet,
    # These ops' PT decomps produce IR that stablehlo can't lower
    # (arange+fmod+index_select, scatter, etc.) — keep C++ decomps instead
    torch.ops.aten.roll,
    torch.ops.aten.one_hot,
    torch.ops.aten.empty_like,
    torch.ops.aten.all,
    torch.ops.aten.isfinite,
    torch.ops.aten.logaddexp,
    torch.ops.aten.logaddexp2,
    torch.ops.aten.eye,
    torch.ops.aten.isclose,
]


def get_expanded_decomposition_table():
    """Use core_aten_decompositions + the extra overloads torch.export produces.

    Excludes decompositions that produce ops torch-mlir doesn't yet support
    (e.g., _fft_r2c) or that regress on complex-number paths.
    """
    from torch._decomp import core_aten_decompositions

    # Each exclude entry is an op (e.g. aten.norm) and drops all of its
    # overloads (aten.norm.Scalar, aten.norm.ScalarOpt_dim, ...).
    # k._overloadpacket is the op that overload k belongs to.
    exclude = set(_EXPANDED_DECOMP_EXCLUDE)
    table = {
        k: v
        for k, v in core_aten_decompositions().items()
        if k._overloadpacket not in exclude
    }
    # Add overloads not in core at the specific overload level needed by torch.export
    table.update(get_decompositions(DEFAULT_DECOMPOSITIONS))
    return table
