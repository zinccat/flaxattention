# flaxattention

Porting [FlexAttention](https://github.com/pytorch-labs/attention-gym) to pure JAX.

Example usage:

```python
import jax
import jax.numpy as jnp
from jax import Array

from flaxattention import flax_attention
from flaxattention.masks import causal_mask
from flaxattention.mods import generate_alibi_bias

if __name__ == "__main__":
    # Prepare inputs
    batch_size = 8
    num_heads = 8
    seq_len_q = 2048
    seq_len_kv = 2048
    feature_size = 64

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
        causal_mask, batch_size, num_heads, seq_len_q, seq_len_kv
    )

    output = flax_attention(
        query,
        key,
        value,
        score_mod=generate_alibi_bias(num_heads),
        block_mask=block_mask,
    )

    print(output.shape)
```

## Installation

```bash
git clone https://github.com/zinccat/flaxattention
cd flaxattention
poetry install
```

## Benchmark

For the checkboard mod_score examples of flexattention, using RTX 3090, with parameters `batch_size=8, num_heads=8, seq_len_q=2048, seq_len_kv=2048, feature_size=64`, under float32, running 100 iterations:

- FlexAttention: 0.82s
- FlaxAttention (This repo): 0.87s
- FlaxAttention (without matmul flag): 0.90s
- FlaxAttention (without pallas softmax): 1.04s

We can see that the performance is about 5% slower than the original implementation. There are still some optimizations to be done.

Another caveat is that Jax default to tf32 for matmul, yet Torch's default is float32, and Torch's tf32 is much faster (0.28s). We can see that the performance is about 3 times slower than the original implementation.

## Issues
Attention under float16 is way too slow due to unoptimized matmul.