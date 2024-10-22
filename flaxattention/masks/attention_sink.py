"""Attention Sink in Efficient Streaming Language Models with Attention Sinks (https://arxiv.org/abs/2309.17453)"""

from flaxattention import _mask_mod_signature, or_masks, and_masks
from flaxattention.masks import causal_mask

def generate_attention_sink(window_size: int, sink_size: int = 4) -> _mask_mod_signature:
    """Generates an attention sink mask with a given window size and sink size.
    Args:
        window_size: The size of the sliding window.
        sink_size: The size of the attention sink.

    Note:
        We assume that the window size represents the lookback size and we mask out all future tokens
        similar to causal masking.
    """
    def sliding_window(b, h, q_idx, kv_idx):
        return q_idx - kv_idx <= window_size
    
    def attention_sink(b, h, q_idx, kv_idx):
        return kv_idx <= sink_size

    attention_sink_mask = and_masks(or_masks(attention_sink, sliding_window), causal_mask)
    attention_sink_mask.__name__ = f"attention_sink_{window_size}_{sink_size}"
    return attention_sink_mask

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

    visualize_attention_scores(query, key, mask_mod=generate_attention_sink(32, 4), name="attention_sink")

if __name__ == "__main__":
    try:
        from jsonargparse import CLI
    except ImportError:
        raise ImportError("Be sure to run: pip install -e .'[viz]'")
    CLI(main)