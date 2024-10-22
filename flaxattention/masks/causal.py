"""Standard Causal Attention Masking."""


def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

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

    visualize_attention_scores(query, key, mask_mod=causal_mask, name="causal_mask")

if __name__ == "__main__":
    try:
        from jsonargparse import CLI
    except ImportError:
        raise ImportError("Be sure to run: pip install -e .'[viz]'")
    CLI(main)