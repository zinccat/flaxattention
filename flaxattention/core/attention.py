import jax
import jax.numpy as jnp
from typing import Any, Callable, Dict, Tuple, Optional, Union, List
from jax import Array

def _vmap_for_bhqkv(
    fn: Callable,
    prefix: Tuple[Optional[int], ...],
    suffix: Tuple[Optional[int], ...] = (),
    out_axes: Union[int, Tuple[Optional[int], ...]] = 0,
    group_dim: bool = False,
) -> Callable:
    """
    Used to vmap both score_mods and mask_mods over 4-dimensional/5-dimensional inputs.
    Mapping over the [b, hq, q_idx, kv_idx] or [b, hkv, g, q_idx, kv_idx] dimensions.

    Args:
        fn (Callable): The function to vmap.
        prefix (Tuple): The prefix of the vmap. For score_mod functions, this should be set to (0,). For mask_mods, use ().
        suffix (Tuple): Additional None entries for other buffers or arguments.
        out_axes (Union[int, Tuple[Optional[int], ...]]): The output axes for the vmapped function.
        group_dim (bool): Whether to include the group dimension.

    Returns:
        Callable: The vmapped function.
    """
    dimensions: List[Tuple[None | int, None | int, None | int, None | int]] = []
    dimensions = [
        (None, None, None, 0),  # Map over kv_idx
        (None, None, 0, None),  # Map over q_idx
        (None, 0, None, None),  # Map over h
    ]

    if group_dim:
        dimensions += [
            (None, 0, None, None),  # Map over group dimension
        ]

    dimensions += [
        (0, None, None, None),  # Map over batch dimension
    ]

    for dims in dimensions:
        in_axes = prefix + dims + suffix
        fn = jax.vmap(fn, in_axes=in_axes, out_axes=out_axes)
    return fn


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

    return scores, post_mod_scores


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
