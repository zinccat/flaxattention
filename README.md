# flaxattention

Porting [FlexAttention](https://github.com/pytorch-labs/attention-gym) to pure JAX.

Example usage (**For faster performance using Flash Attention, check examples/example.py**):

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

    alibi = generate_alibi_bias(num_heads)

    output = flax_attention(
        query,
        key,
        value,
        score_mod=alibi,
        block_mask=block_mask,
    )

    # Using Pallas Flash Attention (Much Faster!)
    output = flax_attention_pallas(
        query,
        key,
        value,
        score_mod=alibi,
        block_mod=causal_mask
    )

    print(output.shape)
    # (8, 8, 2048, 64)
```

## Installation

```bash
git clone https://github.com/zinccat/flaxattention
cd flaxattention
poetry install
```

## Benchmark

For the checkboard mod_score examples of flexattention, using RTX 3090, with parameters `batch_size=8, num_heads=8, seq_len_q=2048, seq_len_kv=2048, feature_size=64`, running 100 iterations:

TF32:
- FlexAttention: 0.27s
- FlaxAttention (This repo): 0.33s
- FlaxAttention (Without Pallas Flash Attention): 0.87s

Float16:
- FlexAttention: 0.11s
- FlaxAttention (This repo): 0.13s

We can see that the performance is about 20% slower than the original implementation. There are still some optimizations to be done.
