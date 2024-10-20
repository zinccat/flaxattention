# Copyright 2023 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module containing fused attention forward and backward pass."""

from __future__ import annotations

import functools
from typing import Any, Callable

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
import jax.numpy as jnp
import numpy as np

from flaxattention.core.common import _score_mod_signature, _mask_mod_signature

DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)


def mha_forward_kernel(
    q_ref,
    k_ref,
    v_ref,  # Input arrays
    segment_ids_ref: jax.Array | None,  # segment_id arrays
    o_ref: Any,  # Output
    *residual_refs: Any,  # Residual outputs
    num_heads: int,
    sm_scale: float,
    causal: bool,
    block_q: int,
    block_d: int,
    block_k: int,
    score_mod: _score_mod_signature | None = None,
    mask_mod: _mask_mod_signature | None = None,
):
    seq_len = k_ref.shape[0]
    start_q = pl.program_id(0)
    start_b = pl.program_id(1)
    start_h = pl.program_id(2)

    # o is the buffer where we accumulate the output on sram.
    # m_i and l_i (see FlashAttention paper) are updated during the k,v loop.
    m_i = jnp.zeros(block_q, dtype=jnp.float32) - float("inf")
    l_i = jnp.zeros(block_q, dtype=jnp.float32)
    # acc is the buffer where we accumulate the output on sram.
    o = jnp.zeros((block_q, block_d), dtype=jnp.float32)

    # Load q: it will stay in L1 throughout. Indices form a matrix because we
    # read, compute, and write all in 2d chunks. 1 element ~= 1 CUDA thread index.
    # q tile has shape [block_q, block_d], block_d == head_dim.
    curr_q_slice = pl.dslice(start_q * block_q, block_q)
    q = q_ref[...]
    q_segment_ids = (
        None if segment_ids_ref is None else pl.load(segment_ids_ref, (curr_q_slice,))
    )

    # In FlashAttention algorithm 1 there are 2 loops: slow over tiles of kv (size
    # (Bc == block_k here), and fast over blocks of q (size Br == block_q here).
    # Here we only loop over blocks of kv to process entire seq_len, the loop over
    # blocks of q is carried out by the grid.
    def body(start_k, carry):
        o_prev, m_prev, l_prev = carry
        curr_k_slice = pl.dslice(start_k * block_k, block_k)

        k = pl.load(k_ref, (curr_k_slice, slice(None)))
        qk = pl.dot(q, k.T)  # [block_q, block_k]
        if sm_scale != 1.0:
            qk *= sm_scale  # [block_q, block_k]

        if score_mod is not None or mask_mod is not None:
            # Apply the custom score modification function here
            span_q = start_q * block_q + jnp.arange(block_q)
            span_k = start_k * block_k + jnp.arange(block_k)
            if mask_mod is not None:
                mask = mask_mod(start_b, start_h, span_q, span_k)
                qk = jnp.where(mask, qk, DEFAULT_MASK_VALUE)
            if score_mod is not None:
                qk = jnp.where(
                    qk != DEFAULT_MASK_VALUE,
                    score_mod(qk, start_b, start_h, span_q, span_k),
                    DEFAULT_MASK_VALUE,
                )
        # Avoids Triton crash.
        # if num_heads > 2:
        #   qk = qk.astype(q_ref.dtype)
        #   qk = qk.astype(jnp.float32)

        if causal or segment_ids_ref is not None:
            mask = None
            if segment_ids_ref is not None:
                kv_segment_ids = pl.load(segment_ids_ref, (curr_k_slice,))
                mask = segment_mask(q_segment_ids, kv_segment_ids)
            if causal:
                span_q = start_q * block_q + jnp.arange(block_q)
                span_k = start_k * block_k + jnp.arange(block_k)
                causal_mask = span_q[:, None] >= span_k[None, :]
                mask = (
                    causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
                )
            # Apply mask to qk.
            qk = jnp.where(mask, qk, DEFAULT_MASK_VALUE)

        m_curr = qk.max(axis=-1)
        m_next = jnp.maximum(m_prev, m_curr)
        correction = jnp.exp(m_prev - m_next)
        l_prev_corr = correction * l_prev
        s_curr = jnp.exp(
            qk - m_next[:, None]
        )  # Use m_next instead of m_curr to avoid a correction on l_curr
        l_curr = s_curr.sum(axis=-1)
        l_next = l_prev_corr + l_curr
        o_prev_corr = correction[:, None] * o_prev
        v = pl.load(v_ref, (curr_k_slice, pl.dslice(block_d)))
        o_curr = pl.dot(s_curr.astype(v.dtype), v)

        o_next = o_prev_corr + o_curr
        return o_next, m_next, l_next

    if causal:
        # Ceildiv (`pl.cdiv` and `//` do not work due to type of start_q)
        upper_bound = lax.div(block_q * (start_q + 1) + block_k - 1, block_k)
    else:
        upper_bound = pl.cdiv(seq_len, block_k)
    o, m_i, l_i = lax.fori_loop(0, upper_bound, body, (o, m_i, l_i))

    # We keep an unscaled version of o during the scan over seq_len. Scaling it
    # by the last l_i gives us the correct final output. See section 3.1.1 in the
    # FlashAttention-2 paper: https://arxiv.org/pdf/2307.08691.
    o /= l_i[:, None]

    if residual_refs:
        lse_ref = residual_refs[0]
        lse_ref[...] = m_i + jnp.log(l_i)
    # Write output to dram.
    o_ref[...] = o.astype(o_ref.dtype)


def segment_mask(
    q_segment_ids: jax.Array,
    kv_segment_ids: jax.Array,
):
    # [B, T, 1] or [T, 1]
    q_segment_ids = jnp.expand_dims(q_segment_ids, axis=-1)
    # [B, 1, S] or [1, S]
    if kv_segment_ids.ndim == 1:
        kv_segment_ids = jnp.expand_dims(kv_segment_ids, axis=0)
    else:
        kv_segment_ids = jnp.expand_dims(kv_segment_ids, axis=1)
    return jnp.equal(q_segment_ids, kv_segment_ids).astype(jnp.bool_)


@functools.partial(
    jax.jit,
    static_argnames=[
        "sm_scale",
        "causal",
        "block_q",
        "block_k",
        "backward_pass_impl",
        "num_warps",
        "num_stages",
        "grid",
        "interpret",
        "debug",
        "score_mod",
        "mask_mod",
    ],
)
def mha(
    q,
    k,
    v,
    segment_ids: jnp.ndarray | None,
    sm_scale: float = 1.0,
    causal: bool = False,
    block_q: int = 128,
    block_k: int = 128,
    backward_pass_impl: str = "triton",
    num_warps: int | None = None,
    num_stages: int = 2,
    grid: tuple[int, ...] | None = None,
    interpret: bool = False,
    debug: bool = False,
    score_mod: _score_mod_signature | None = None,
    mask_mod: _mask_mod_signature | None = None,
):
    del backward_pass_impl
    batch_size, seq_len, num_heads, head_dim = q.shape
    block_q = min(block_q, seq_len)
    block_k = min(block_k, seq_len)
    # Heuristics.
    grid_ = grid
    if grid_ is None:
        grid_ = (pl.cdiv(seq_len, block_q), batch_size, num_heads) # seq, batch, head

    num_warps_ = num_warps
    if num_warps_ is None:
        num_warps_ = 4 if head_dim <= 64 else 8
    kernel = functools.partial(
        mha_forward_kernel,
        num_heads=num_heads,
        sm_scale=sm_scale,
        block_q=block_q,
        block_k=block_k,
        block_d=head_dim,
        causal=causal,
        score_mod=score_mod,
        mask_mod=mask_mod,
    )

    in_specs = [
        pl.BlockSpec((None, block_q, None, head_dim), lambda i, j, k: (j, i, k, 0)),
        pl.BlockSpec((None, seq_len, None, head_dim), lambda _, j, k: (j, 0, k, 0)),
        pl.BlockSpec((None, seq_len, None, head_dim), lambda _, j, k: (j, 0, k, 0)),
    ]
    in_specs.append(
        None  # type: ignore[arg-type]
        if segment_ids is None
        else pl.BlockSpec((None, seq_len), lambda _, j, k: (j, 0))
    )
    out_shape = jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype)
    return pl.pallas_call(
        kernel,
        grid=grid_,
        in_specs=in_specs,
        out_specs=pl.BlockSpec(
            (None, block_q, None, head_dim), lambda i, j, k: (j, i, k, 0)
        ),
        compiler_params=plgpu.TritonCompilerParams(
            num_warps=num_warps_, num_stages=num_stages
        ),
        out_shape=out_shape,
        debug=debug,
        interpret=interpret,
        name="mha_forward",
    )(q, k, v, segment_ids)
