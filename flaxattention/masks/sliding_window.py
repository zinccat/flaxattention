"""Generates a sliding window attention mask"""

from flaxattention import _mask_mod_signature, and_masks
from flaxattention.masks import causal_mask


def generate_sliding_window(window_size: int) -> _mask_mod_signature:
    """Generates a sliding window attention mask with a given window size.
    Args:
        window_size: The size of the sliding window.

    Note:
        We assume that the window size represents the lookback size and we mask out all future tokens
        similar to causal masking.
    """

    def sliding_window(b, h, q_idx, kv_idx):
        return q_idx - kv_idx <= window_size

    sliding_window_mask = and_masks(sliding_window, causal_mask)
    sliding_window_mask.__name__ = f"sliding_window_{window_size}"
    return sliding_window_mask

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

    visualize_attention_scores(query, key, mask_mod=generate_sliding_window(32), name="sliding_window")

if __name__ == "__main__":
    try:
        from jsonargparse import CLI
    except ImportError:
        raise ImportError("Be sure to run: pip install -e .'[viz]'")
    CLI(main)