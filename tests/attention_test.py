import jax
from jax import Array
import jax.numpy as jnp
import torch
import torch.utils.dlpack
from torch.nn.attention.flex_attention import flex_attention
import numpy as np
from absl.testing import absltest # pylint: disable=import-error

from flaxattention import flax_attention, flax_attention_pallas, create_block_mask
from flaxattention.mods import generate_alibi_bias


def jax2torch(array: Array) -> torch.Tensor:
    return torch.from_dlpack((jax.dlpack.to_dlpack(array)))


def torch2jax(tensor: torch.Tensor) -> Array:
    return jax.dlpack.from_dlpack(torch.to_dlpack(tensor))


class TestAttention(absltest.TestCase):
    def test_equivalence_with_torch(self):
        # Prepare inputs
        batch_size = 4
        num_heads = 8
        seq_len_q = 64
        seq_len_kv = 64
        feature_size = 32

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

        output_jax = flax_attention(
            query,
            key,
            value,
        )

        query_torch = jax2torch(query)
        key_torch = jax2torch(key)
        value_torch = jax2torch(value)

        output_torch = (
            flex_attention(
                query_torch,
                key_torch,
                value_torch,
            )
            .detach()
            .cpu()
            .numpy()
        )

        np.testing.assert_almost_equal(output_jax, output_torch, decimal=2)

    def test_fully_masked(self):
        # Prepare inputs
        batch_size = 4
        num_heads = 8
        seq_len_q = 64
        seq_len_kv = 64
        feature_size = 32

        def fullmask(batch_idx, head_idx, q_idx, kv_idx):
            return False
        
        block_mask = create_block_mask(fullmask, batch_size, num_heads, seq_len_q, seq_len_kv)

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

        output = flax_attention(
            query,
            key,
            value,
            block_mask=block_mask,
        )

        np.testing.assert_almost_equal(output, np.zeros_like(output), decimal=2)

    def test_pallas_fully_masked(self):
        # Prepare inputs
        batch_size = 4
        num_heads = 8
        seq_len_q = 64
        seq_len_kv = 64
        feature_size = 32

        def fullmask(batch_idx, head_idx, q_idx, kv_idx):
            return False
        
        # block_mask = create_block_mask(fullmask, batch_size, num_heads, seq_len_q, seq_len_kv)

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

        output = flax_attention_pallas(
            query,
            key,
            value,
            mask_mod=fullmask,
        )

        np.testing.assert_almost_equal(output, np.zeros_like(output), decimal=2)

    def test_pallas_equivalence_with_torch(self):
        # Prepare inputs
        batch_size = 4
        num_heads = 8
        seq_len_q = 64
        seq_len_kv = 64
        feature_size = 32

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

        output_jax = flax_attention_pallas(
            query,
            key,
            value,
        )

        query_torch = jax2torch(query)
        key_torch = jax2torch(key)
        value_torch = jax2torch(value)

        output_torch = (
            flex_attention(
                query_torch,
                key_torch,
                value_torch,
            )
            .detach()
            .cpu()
            .numpy()
        )

        np.testing.assert_almost_equal(output_jax, output_torch, decimal=2)
    
    def test_decoding_equivalence_with_torch(self):
        # Prepare inputs
        batch_size = 4
        num_heads = 8
        seq_len_q = 1
        seq_len_kv = 64
        feature_size = 32

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

        output_jax = flax_attention(
            query,
            key,
            value,
            score_mod=generate_alibi_bias(num_heads),
        )

        query_torch = jax2torch(query)
        key_torch = jax2torch(key)
        value_torch = jax2torch(value)

        def generate_alibi_bias_torch(H: int):
            """Returns an alibi bias score_mod given the number of heads H

            Args:
                H: number of heads

            Returns:
                alibi_bias: alibi bias score_mod
            """

            def alibi_mod(score, b, h, q_idx, kv_idx):
                scale = torch.exp2(-((h + 1) * 8.0 / H))
                bias = (q_idx - kv_idx) * scale
                return score + bias

            return alibi_mod

        output_torch = (
            flex_attention(
                query_torch,
                key_torch,
                value_torch,
                score_mod=generate_alibi_bias_torch(num_heads),
            )
            .detach()
            .cpu()
            .numpy()
        )

        np.testing.assert_almost_equal(output_jax, output_torch, decimal=2)

    def test_pallas_decoding_equivalence_with_torch(self):
        # Prepare inputs
        batch_size = 4
        num_heads = 8
        seq_len_q = 1
        seq_len_kv = 64
        feature_size = 32

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

        output_jax = flax_attention_pallas(
            query,
            key,
            value,
            score_mod=generate_alibi_bias(num_heads),
        )

        query_torch = jax2torch(query)
        key_torch = jax2torch(key)
        value_torch = jax2torch(value)

        def generate_alibi_bias_torch(H: int):
            """Returns an alibi bias score_mod given the number of heads H

            Args:
                H: number of heads

            Returns:
                alibi_bias: alibi bias score_mod
            """

            def alibi_mod(score, b, h, q_idx, kv_idx):
                scale = torch.exp2(-((h + 1) * 8.0 / H))
                bias = (q_idx - kv_idx) * scale
                return score + bias

            return alibi_mod

        output_torch = (
            flex_attention(
                query_torch,
                key_torch,
                value_torch,
                score_mod=generate_alibi_bias_torch(num_heads),
            )
            .detach()
            .cpu()
            .numpy()
        )

        np.testing.assert_almost_equal(output_jax, output_torch, decimal=2)

    def test_gqa(self):
        # Prepare inputs
        batch_size = 4
        num_heads_q = 8
        num_heads_kv = 2
        seq_len_q = 64
        seq_len_kv = 64
        feature_size = 32

        # Random tensors for query, key, and value
        key = jax.random.normal(
            jax.random.PRNGKey(0), (batch_size, num_heads_kv, seq_len_kv, feature_size)
        )
        query = jax.random.normal(
            jax.random.PRNGKey(1), (batch_size, num_heads_q, seq_len_q, feature_size)
        )
        value = jax.random.normal(
            jax.random.PRNGKey(2), (batch_size, num_heads_kv, seq_len_kv, feature_size)
        )

        output = flax_attention(
            query,
            key,
            value,
            enable_gqa=True,
        )

        self.assertEqual(
            output.shape, (batch_size, num_heads_q, seq_len_q, feature_size)
        )

    def test_score_mod(self):
        # Prepare inputs
        batch_size = 4
        num_heads = 8
        seq_len_q = 64
        seq_len_kv = 64
        feature_size = 32

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

        output = flax_attention(
            query,
            key,
            value,
            score_mod=generate_alibi_bias(num_heads),
        )

        self.assertEqual(output.shape, (batch_size, num_heads, seq_len_q, feature_size))

    def test_autograd(self):
        # Prepare inputs
        batch_size = 4
        num_heads = 8
        seq_len_q = 64
        seq_len_kv = 64
        feature_size = 32

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

        def fn(query, key, value):
            return flax_attention(
                query,
                key,
                value,
            ).sum()
        
        grad_fn = jax.grad(fn, 0)
        grad = grad_fn(query, key, value)

        self.assertEqual(grad.shape, (batch_size, num_heads, seq_len_q, feature_size))

    def test_autograd_equivalence_with_torch(self):
        # Prepare inputs
        batch_size = 4
        num_heads = 8
        seq_len_q = 64
        seq_len_kv = 64
        feature_size = 32

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

        def fn(query, key, value):
            return flax_attention(
                query,
                key,
                value,
            ).sum()
        
        grad_fn = jax.grad(fn, 0)
        grad_jax = grad_fn(query, key, value)

        query_torch = jax2torch(query)
        key_torch = jax2torch(key)
        value_torch = jax2torch(value)

        query_torch.requires_grad = True

        output_torch = flex_attention(
            query_torch,
            key_torch,
            value_torch,
        ).sum()

        output_torch.backward()

        grad_torch = query_torch.grad.cpu().numpy()

        np.testing.assert_almost_equal(grad_jax, grad_torch, decimal=2)

    def test_pallas_autograd_equivalence_with_torch(self):
        # Prepare inputs
        batch_size = 4
        num_heads = 8
        seq_len_q = 64
        seq_len_kv = 64
        feature_size = 32

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

        def fn(query, key, value):
            return flax_attention_pallas(
                query,
                key,
                value,
            ).sum()
        
        grad_fn = jax.grad(fn, 0)
        grad_jax = grad_fn(query, key, value)

        query_torch = jax2torch(query)
        key_torch = jax2torch(key)
        value_torch = jax2torch(value)

        query_torch.requires_grad = True

        output_torch = flex_attention(
            query_torch,
            key_torch,
            value_torch,
        ).sum()

        output_torch.backward()

        grad_torch = query_torch.grad.cpu().numpy()

        np.testing.assert_almost_equal(grad_jax, grad_torch, decimal=2)

    def test_decoding_autograd_equivalence_with_torch(self):
        # Prepare inputs
        batch_size = 4
        num_heads = 8
        seq_len_q = 1
        seq_len_kv = 64
        feature_size = 32

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

        def fn(query, key, value):
            return flax_attention(
                query,
                key,
                value,
            ).sum()
        
        grad_fn = jax.grad(fn, 0)
        grad_jax = grad_fn(query, key, value)

        query_torch = jax2torch(query)
        key_torch = jax2torch(key)
        value_torch = jax2torch(value)

        query_torch.requires_grad = True

        output_torch = flex_attention(
            query_torch,
            key_torch,
            value_torch,
        ).sum()

        output_torch.backward()

        grad_torch = query_torch.grad.cpu().numpy()

        np.testing.assert_almost_equal(grad_jax, grad_torch, decimal=2)