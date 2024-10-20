"""Standard Causal Attention Masking."""


def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx
