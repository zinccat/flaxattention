from torch.nn.attention.flex_attention import flex_attention, create_block_mask
import torch
import numpy as np

if __name__ == "__main__":
    batch_size = 8
    num_heads = 8
    seq_len_q = 1
    seq_len_kv = 2048
    feature_size = 64

    query_torch = torch.randn(batch_size, num_heads, seq_len_q, feature_size, dtype=torch.float16).cuda()
    key_torch = torch.randn(batch_size, num_heads, seq_len_kv, feature_size, dtype=torch.float16).cuda()
    value_torch = torch.randn(batch_size, num_heads, seq_len_kv, feature_size, dtype=torch.float16).cuda()

    query_torch.requires_grad = True

    def checkerboard_torch(score, batch, head, q_idx, k_idx):
        score = torch.where((k_idx - q_idx) % 2 == 0, score * 0.5, score)
        score = torch.where((k_idx - q_idx) % 2 == 1, score * 2.0, score)
        return score
    
    flex_attention = torch.compile(flex_attention)
    
    # warmup
    
    output_torch = flex_attention(
            query_torch,
            key_torch,
            value_torch,
            score_mod=checkerboard_torch,
    )

    # benchmark
    from timeit import default_timer as timer
    start = timer()
    for _ in range(100):
        output_torch = flex_attention(
            query_torch,
            key_torch,
            value_torch,
            score_mod=checkerboard_torch,
        )
    torch.cuda.synchronize()
    end = timer()

    print("Time taken for 100 iterations: ", end - start)

    start = timer()
    for _ in range(100):
        output_torch = flex_attention(
            query_torch,
            key_torch,
            value_torch,
            score_mod=checkerboard_torch,
        ).sum().backward()
    torch.cuda.synchronize()
    end = timer()

    print("Time taken for 100 iterations backprop: ", end - start)