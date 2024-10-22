from jax import numpy as jnp
from flaxattention import _score_mod_signature


def generate_alibi_bias(H: int) -> _score_mod_signature:
    """Returns an alibi bias score_mod given the number of heads H

    Args:
        H: number of heads

    Returns:
        alibi_bias: alibi bias score_mod
    """

    def alibi_mod(score, b, h, q_idx, kv_idx):
        scale = jnp.exp2(-((h + 1) * 8.0 / H))
        bias = (q_idx - kv_idx) * scale
        return score + bias

    return alibi_mod

def main(device: str = "cpu"):
    """Visualize the attention scores of causal masking.

    Args:
        device (str): Device to use for computation. Defaults
    """
    from flaxattention.utils import visualize_attention_scores
    import jax.numpy as jnp

    B, H, SEQ_LEN, HEAD_DIM = 1, 1, 128, 8

    def make_tensor():
        return jnp.ones((B, H, SEQ_LEN, HEAD_DIM))

    query, key = make_tensor(), make_tensor()

    visualize_attention_scores(query, key, score_mod=generate_alibi_bias(H), name="alibi_bias")

if __name__ == "__main__":
    try:
        from jsonargparse import CLI
    except ImportError:
        raise ImportError("Be sure to run: pip install -e .'[viz]'")
    CLI(main)
