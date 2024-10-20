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
