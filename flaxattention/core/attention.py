import math
import jax
import jax.numpy as jnp
from typing import Any, Callable, Dict, Tuple, Optional, Union
from jax import Array

from .common import _score_mod_signature, _vmap_for_bhqkv
from .blockmask import _create_empty_block_mask, BlockMask


def _math_attention_inner(
    query: Array,
    key: Array,
    value: Array,
    score_mod: Callable,
    block_mask: Tuple,
    scale: float,
    kernel_options: Dict[str, Any],
    score_mod_other_buffers: Tuple = (),
    mask_mod_other_buffers: Tuple = (),
) -> Tuple[Array, Array]:
    working_precision = jnp.float64 if query.dtype == jnp.float64 else jnp.float32

    scores = jnp.matmul(query, jnp.swapaxes(key, -2, -1)).astype(working_precision)

    b_size, h_size, m_size, n_size = scores.shape
    b_idx = jnp.arange(b_size)
    h_idx = jnp.arange(h_size)
    m_idx = jnp.arange(m_size)
    n_idx = jnp.arange(n_size)

    captured_buffers_in_dim = (None,) * len(score_mod_other_buffers)
    score_mod = _vmap_for_bhqkv(
        score_mod,
        prefix=(0,),  # The first argument (score) is mapped over batch dimension
        suffix=captured_buffers_in_dim,
        out_axes=0,
        group_dim=False,
    )

    mask_mod_in_dim_buffers = (None,) * len(mask_mod_other_buffers)
    mask_mod = block_mask[-1]
    mask_mod = _vmap_for_bhqkv(
        mask_mod,
        prefix=(),
        suffix=mask_mod_in_dim_buffers,
        out_axes=0,
        group_dim=False,
    )

    # Scaling
    scores = (scores * scale).astype(working_precision)

    mask = mask_mod(b_idx, h_idx, m_idx, n_idx, *mask_mod_other_buffers)
    mask = mask.astype(jnp.bool_)
    post_mod_scores = jnp.where(
        mask,
        score_mod(scores, b_idx, h_idx, m_idx, n_idx, *score_mod_other_buffers),
        -jnp.inf,
    )

    return scores, post_mod_scores  # type: ignore


def math_attention(
    query: Array,
    key: Array,
    value: Array,
    score_mod: Callable,
    block_mask: Tuple,
    scale: float,
    kernel_options: Dict[str, Any],
    score_mod_other_buffers: Tuple = (),
    mask_mod_other_buffers: Tuple = (),
) -> Tuple[Array, Array]:
    # Broadcast query & key along head dimension for GQA
    G = query.shape[1] // key.shape[1]
    value = jnp.repeat(value, G, axis=1)
    key = jnp.repeat(key, G, axis=1)

    # Compute modified scores
    _, post_mod_scores = _math_attention_inner(
        query,
        key,
        value,
        score_mod,
        block_mask,
        scale,
        kernel_options,
        score_mod_other_buffers,
        mask_mod_other_buffers,
    )

    logsumexp = jax.scipy.special.logsumexp(post_mod_scores, axis=-1)
    masked_rows = jnp.all(post_mod_scores == -jnp.inf, axis=-1)
    logsumexp = jnp.where(masked_rows, -jnp.inf, logsumexp)

    post_mod_scores = jax.nn.softmax(post_mod_scores, axis=-1)

    output = jnp.matmul(post_mod_scores.astype(query.dtype), value)
    return output, logsumexp / jnp.log(2)


def _identity(
    score: Array,
    batch: Array,
    head: Array,
    token_q: Array,
    token_kv: Array,
) -> Array:
    return score


def _validate_sdpa_input(
    query: Array,
    key: Array,
    value: Array,
    attn_mask: Optional[Array] = None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
):
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError(
            f"Expected query, key, and value to have the same dtype, "
            f"but got query.dtype: {query.dtype}, key.dtype: {key.dtype}, "
            f"and value.dtype: {value.dtype} instead."
        )
    if query.ndim < 2 or key.ndim < 2 or value.ndim < 2:
        raise ValueError(
            f"Expected query, key, and value to all be  at least 2 dimensional, but got query.dim: "
            f"{query.ndim}, key.dim: {key.ndim} and value.dim: {value.ndim} instead."
        )


_SUPPORTED_HEAD_DIMS = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


def _supported_head_dim(n: int) -> bool:
    """Returns true if the head dim is supported by FlexAttention"""
    return n in _SUPPORTED_HEAD_DIMS


def _validate_embed_dim(query: Array, key: Array, value: Array):
    if query.shape[-1] != key.shape[-1]:
        raise ValueError(
            f"Expect query and key/value to have the same embedding dimension "
            f"but got E={query.shape[-1]} and E={key.shape[-1]}."
        )
    # Not sure if this is necessary
    # if not (
    #     _supported_head_dim(query.shape[-1]) and _supported_head_dim(value.shape[-1])
    # ):
    #     raise ValueError(
    #         f"NYI: Currently non power of 2 embedding dimension are not supported. "
    #         f"Got E={query.shape[-1]} and Ev={value.shape[-1]}."
    #     )
    if value.shape[-1] > query.shape[-1]:
        raise ValueError(
            f"NYI: Currently value embedding dimension must be less than or equal to query embedding dimension. "
            f"Got Ev={value.shape[-1]} and E={query.shape[-1]}."
        )


def flax_attention(
    query: Array,
    key: Array,
    value: Array,
    score_mod: Optional[_score_mod_signature] = None,
    block_mask: Optional[BlockMask] = None,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
    return_lse: bool = False,
    kernel_options: Optional[Dict[str, Any]] = None,
) -> Union[Array, Tuple[Array, Array]]:
    _validate_sdpa_input(query, key, value)
    _validate_embed_dim(query, key, value)
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise NotImplementedError("NYI: query, key, and value must be 4D tensors")
    if (not enable_gqa) and query.shape[-3] != key.shape[-3]:
        raise ValueError(
            f"Expect query and key/value to have the same number of heads "
            f"but got Hq={query.shape[-3]} and Hkv={key.shape[-3]}. "
            f"Try setting enable_gqa=True for GQA."
        )
    if enable_gqa:
        Hq = query.shape[1]
        Hkv = key.shape[1]
        if Hq % Hkv != 0:
            raise ValueError(
                f"Expect number of query heads to be a multiple of kv heads for GQA "
                f"but got Hq={Hq} and Hkv={Hkv}."
            )

    if score_mod is None:
        score_mod = _identity
    if block_mask is None:
        block_mask = _create_empty_block_mask(query, key)
    if scale is None:
        scale = 1.0 / math.sqrt(query.shape[-1])
    out, lse = math_attention(
        query,
        key,
        value,
        score_mod,
        block_mask.as_tuple(),
        scale,
        kernel_options,  # type: ignore
    )
    if return_lse:
        return out, lse * jnp.log(2)
    else:
        return out
