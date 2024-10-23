import jax
import jax.numpy as jnp
from jax import Array
from jax.nn import dot_product_attention

from flaxattention import flax_attention, create_block_mask, flax_attention_pallas

import os

os.environ["XLA_FLAGS"] = (
    "--xla_gpu_enable_triton_softmax_fusion=true " "--xla_gpu_triton_gemm_any=True"
)

if __name__ == "__main__":

    @jax.jit
    def checkerboard(
        score: Array, batch: Array, head: Array, q_idx: Array, k_idx: Array
    ) -> Array:
        score = jnp.where((k_idx - q_idx) % 2 == 0, score * 0.5, score)
        score = jnp.where((k_idx - q_idx) % 2 == 1, score * 2.0, score)
        return score

    @jax.jit
    def causal(batch: Array, head: Array, q_idx: Array, k_idx: Array) -> Array:
        return q_idx >= k_idx

    # Prepare inputs
    batch_size = 8
    num_heads = 8
    seq_len_q = 1
    seq_len_kv = 2048
    feature_size = 64

    # Random tensors for query, key, and value
    key = jax.random.normal(
        jax.random.PRNGKey(0),
        (batch_size, num_heads, seq_len_kv, feature_size),
        dtype=jnp.float16,
    )
    query = jax.random.normal(
        jax.random.PRNGKey(1),
        (batch_size, num_heads, seq_len_q, feature_size),
        dtype=jnp.float16,
    )
    value = jax.random.normal(
        jax.random.PRNGKey(2),
        (batch_size, num_heads, seq_len_kv, feature_size),
        dtype=jnp.float16,
    )

    block_mask = create_block_mask(causal, batch_size, num_heads, seq_len_q, seq_len_kv)

    output = flax_attention(
        query,
        key,
        value,
        score_mod=checkerboard,
        # block_mask=block_mask,
    )
    output.block_until_ready()
    # print(output[0, 0, 0])

    # benchmark
    from timeit import default_timer as timer

    start = timer()
    for _ in range(100):
        output = flax_attention(
            query,
            key,
            value,
            score_mod=checkerboard,
            # block_mask=block_mask,
        )
    output.block_until_ready()
    end = timer()
    print("Pure jax time taken:", end - start)

    # try mha pallas kernel
    
    # warm up
    output = flax_attention_pallas(
        query,
        key,
        value,
        score_mod=checkerboard,
        # mask_mod=causal,
    )
    output.block_until_ready()
    # print(output[0, 0, 0])

    start = timer()
    for _ in range(100):
        output = flax_attention_pallas(
            query,
            key,
            value,
            score_mod=checkerboard,
            # mask_mod=causal,
        )
    output.block_until_ready()
    end = timer()
    print("Pallas attention time taken:", end - start)

    # try jax attention

    query_transposed = jnp.moveaxis(query, 1, 2)
    key_transposed = jnp.moveaxis(key, 1, 2)
    value_transposed = jnp.moveaxis(value, 1, 2)
    # warm up
    output = dot_product_attention(
        query_transposed,
        key_transposed,
        value_transposed,
    )
    output.block_until_ready()

    start = timer()
    for _ in range(100):
        output = dot_product_attention(
            query_transposed,
            key_transposed,
            value_transposed,
        )
    output.block_until_ready()
    end = timer()
    print("Jax dot product attention time taken (no score_mod):", end - start)

    # original palllas attention
    from jax.experimental.pallas.ops.gpu.decode_attention import gqa

    query = query.squeeze(2)
    key = jnp.moveaxis(key, 1, 2)
    value = jnp.moveaxis(value, 1, 2)

    output = gqa(query, key, value)
    output.block_until_ready()

    start = timer()
    for _ in range(100):
        output = gqa(query, key, value)
    end = timer()
    output.block_until_ready()
    print("Original Pallas decoding attention time taken:", end - start)

    # original palllas attention
    from jax.experimental.pallas.ops.gpu.attention import mha

    query = query[:, :, None, :]

    # pad query
    if query.shape[1] < 16:
        pad = 16 - query.shape[1]
        query = jnp.pad(query, ((0, 0), (0, pad), (0, 0), (0, 0)))

    output = mha(query, key, value, segment_ids=None, block_q=64, block_k=64)
    output.block_until_ready()

    start = timer()
    for _ in range(100):
        output = mha(query, key, value, segment_ids=None)
    end = timer()
    output.block_until_ready()
    print("Original Pallas attention time taken:", end - start)