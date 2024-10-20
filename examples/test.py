import jax
import jax.numpy as jnp
from jax import Array

from flaxattention import flax_attention, create_block_mask

if __name__ == "__main__":

    def mask_mod(b_idx, h_idx, q_idx, k_idx):
        # return 1 if q_idx >= k_idx else 0
        # all values with 0 are masked
        return q_idx >= k_idx

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
        jax.random.PRNGKey(0), (batch_size, num_heads, seq_len_kv, feature_size)
    )
    query = jax.random.normal(
        jax.random.PRNGKey(1), (batch_size, num_heads, seq_len_q, feature_size)
    )
    value = jax.random.normal(
        jax.random.PRNGKey(2), (batch_size, num_heads, seq_len_kv, feature_size)
    )

    flax_attention = jax.jit(flax_attention, static_argnums=(3, 4))

    block_mask = create_block_mask(
        mask_mod, batch_size, num_heads, seq_len_q, seq_len_kv
    )

    output = flax_attention(
        query,
        key,
        value,
        score_mod=checkerboard,
        block_mask=block_mask,
    )

    # benchmark
    from timeit import default_timer as timer

    start = timer()
    for _ in range(10):
        output = flax_attention(
            query,
            key,
            value,
            score_mod=checkerboard,
            block_mask=block_mask,
        )
    output[0].block_until_ready()
    end = timer()
    print("Time taken:", end - start)

    print("Output shape:", output.shape)
