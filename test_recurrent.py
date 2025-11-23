import torch
import torch.nn.functional as F

from HRDPLR import (
    fused_recurrent_dplr_delta_rule, 
    chunk_hrdplr_delta_rule, 
    hrdplr_naive, 
)

if __name__ == '__main__':
    B = 2
    H = 4
    T = 256
    K = 128
    V = 64
    RANK_AB = 2

    BT = 16

    device = torch.device('cuda')
    dtype = torch.float32

    q = torch.randn(size=(B, T, H, K), device=device, dtype=dtype, requires_grad=True)
    
    k = torch.randn(size=(B, T, H, K), device=device, dtype=dtype)
    k /= (torch.norm(k, dim=-1, keepdim=True) + 1e-6)
    k.requires_grad_()

    v = torch.randn(size=(B, T, H, V), device=device, dtype=dtype, requires_grad=True)
    
    a = torch.randn(size=(B, T * RANK_AB, H, K), device=device, dtype=dtype)
    a /= (torch.norm(a, dim=-1, keepdim=True) + 1e-6)
    a.requires_grad_()

    b = torch.randn(size=(B, T * RANK_AB, H, K), device=device, dtype=dtype)
    b /= (torch.norm(b, dim=-1, keepdim=True) + 1e-6)
    b.requires_grad_()

    gk = F.logsigmoid(
        torch.randn(size=(B, T, H, K), device=device, dtype=torch.float32),
    )

    # For debug
    # gk = torch.zeros(size=(B, T, H, K), device=device, dtype=torch.float32) - 0.04
    # gk.requires_grad_()

    initial_state = torch.randn(size=(B, H, K, V), device=device, dtype=dtype, requires_grad=True)
    o_chunk, final_state_chunk = chunk_hrdplr_delta_rule(
        q, k, v, a, b, gk, RANK_AB, 
        scale=1.0, # equivalent to apply row-wise on q
        initial_state=initial_state, # [B, H, K, V]
        output_final_state=True, 
        head_first=False, 
        chunk_size=BT, 
    )

    o_naive, final_state_naive = hrdplr_naive(
        q, k, v, a, b, gk, 1.0, initial_state, True, 
        RANK_AB, False, BT    
    )

    o_recurrent, final_state_recurrent = fused_recurrent_dplr_delta_rule(
        q, k, v, a, b, gk, RANK_AB, 1.0, initial_state, True, False, None, 
    )

    assert torch.allclose(o_chunk, o_recurrent, rtol=5e-2, atol=5e-2)
    assert torch.allclose(final_state_chunk, final_state_recurrent, rtol=5e-2, atol=5e-2)