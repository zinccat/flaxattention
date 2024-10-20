import math
import jax
import jax.numpy as jnp
from jax import Array
from typing import Callable
from functools import partial

from flaxattention import flax_attention
from flaxattention.kernel.attention import mha as mha_pallas

import os
os.environ['XLA_FLAGS'] = (
    # '--xla_gpu_enable_triton_softmax_fusion=true'
    '--xla_gpu_triton_gemm_any=True'
)

@partial(jax.jit, static_argnums=(3,))
def mha(
    query: Array,
    key: Array,
    value: Array,
    score_mod: Callable,
    block_mask: Array = None,
) -> Array:
    whole = jnp.arange(query.shape[2]).reshape(1, query.shape[2])
    sm_scale = 1 / math.sqrt(query.shape[-1])
    block_q = 64
    block_k = 64
    return mha_pallas(
        query,
        key,
        value,
        whole,
        None,
        sm_scale,
        # block_mask=block_mask,
        block_q=block_q,
        block_k=block_k,
        score_mod=score_mod,
    )

if __name__ == "__main__":

    def checkerboard(
        score: Array, batch: Array, head: Array, q_idx: Array, k_idx: Array
    ) -> Array:
        score = jnp.where((k_idx - q_idx) % 2 == 0, score * 0.5, score)
        score = jnp.where((k_idx - q_idx) % 2 == 1, score * 2.0, score)
        return score

    # Prepare inputs
    batch_size = 8
    num_heads = 8
    seq_len_q = 2048
    seq_len_kv = 2048
    feature_size = 64

    # Random tensors for query, key, and value
    key = jax.random.normal(
        jax.random.PRNGKey(0), (batch_size, num_heads, seq_len_kv, feature_size), #dtype=jnp.float16
    )
    query = jax.random.normal(
        jax.random.PRNGKey(1), (batch_size, num_heads, seq_len_q, feature_size), #dtype=jnp.float16
    )
    value = jax.random.normal(
        jax.random.PRNGKey(2), (batch_size, num_heads, seq_len_kv, feature_size), #dtype=jnp.float16
    )

    flax_attention = jax.jit(flax_attention, static_argnums=(3, 4))

    output = flax_attention(
        query,
        key,
        value,
        score_mod=checkerboard
    )

    # benchmark
    from timeit import default_timer as timer

    start = timer()
    for _ in range(100):
        output = flax_attention(
        query,
        key,
        value,
        score_mod=checkerboard
    )
    output[0].block_until_ready()
    end = timer()
    print("Pure jax time taken:", end - start)

    # try flax attention
    from flax.nnx import dot_product_attention
    start = timer()
    for _ in range(100):
        output = dot_product_attention(
            query,
            key,
            value,
        )
    output.block_until_ready()
    end = timer()
    print("Flax attention time taken (no score_mod):", end - start)

    # try mha kernel
    query = jnp.moveaxis(query, 2, 1)
    key = jnp.moveaxis(key, 2, 1)
    value = jnp.moveaxis(value, 2, 1)
    dimensions = [
        (None, None, None, 0),  # Map over kv_idx
        (None, None, 0, None),  # Map over q_idx
    ]
    prefix=(0,)
    for dims in dimensions:
        in_axes = prefix + dims
        checkerboard = jax.vmap(checkerboard, in_axes=in_axes, out_axes=0)

    start = timer()
    for _ in range(100):
        output = mha(
            query,
            key,
            value,
            score_mod=checkerboard,
        )
    output.block_until_ready()
    end = timer()
    print("Pallas attention time taken:", end - start)
    output = jnp.moveaxis(output, 1, 2)
    print(output.shape)