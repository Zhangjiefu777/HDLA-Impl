import torch
import torch.nn.functional as F

from fla.ops.generalized_delta_rule import (
    chunk_dplr_delta_rule, fused_recurrent_dplr_delta_rule
)

if __name__ == '__main__':
    B = 2
    H = 4
    T = 256
    K = 128
    V = 64

    BT = 32

    device = torch.device('cuda')
    dtype = torch.float32

    q = torch.randn(size=(B, T, H, K), device=device, dtype=dtype, requires_grad=True)
    
    k = torch.randn(size=(B, T, H, K), device=device, dtype=dtype)
    k /= (torch.norm(k, dim=-1, keepdim=True) + 1e-6)
    k.requires_grad_()

    v = torch.randn(size=(B, T, H, V), device=device, dtype=dtype, requires_grad=True)
    
    a = torch.randn(size=(B, T, H, K), device=device, dtype=dtype)
    a /= (torch.norm(a, dim=-1, keepdim=True) + 1e-6)
    a.requires_grad_()

    b = torch.randn(size=(B, T, H, K), device=device, dtype=dtype)
    b /= (torch.norm(b, dim=-1, keepdim=True) + 1e-6)
    b.requires_grad_()

    gk = F.logsigmoid(
        torch.randn(size=(B, T, H, K), device=device, dtype=torch.float32),
    )
    gk.requires_grad_()

    initial_state = torch.randn(size=(B, H, K, V), device=device, dtype=dtype, requires_grad=True)
    o_chunk, final_state_chunk = chunk_dplr_delta_rule(
        q,
        k,
        v,
        a,
        b,
        gk,
        1.0,
        initial_state,
        output_final_state=True,
        cu_seqlens=None,
        head_first=False,
    )

    o_recurrent, final_state_recurrent = fused_recurrent_dplr_delta_rule(
        q,
        k,
        v,
        a,
        b,
        gk,
        1.0,
        initial_state,
        output_final_state=True,
        reverse=False,
        cu_seqlens=None,
    )

    assert torch.allclose(o_chunk, o_recurrent, rtol=5e-2, atol=5e-2)
    assert torch.allclose(final_state_chunk, final_state_recurrent, rtol=5e-2, atol=5e-2)