import triton
import torch

from HRDPLR.bwd_triton import chunk_dplr_bwd_kernel_dAu
from einops import einsum

if __name__ == '__main__':
    B = 4
    H = 4
    T = 1024
    V = 64
    BV = 32

    RANK_AB = 2

    BT = 32
    NT = triton.cdiv(T, BT)
    
    device = torch.device('cuda:0')
    dtype = torch.bfloat16

    grid = (NT, B * H)
    v = torch.randn(
        size=(B, T, H, V), device=device, dtype=dtype, 
    )
    do = torch.randn(
        size=(B, T, H, V), device=device, dtype=dtype, 
    )
    v_new = torch.randn(
        size=(B, T * RANK_AB, H, V), device=device, dtype=dtype, 
    )
    A_qb = torch.randn(
        size=(B, T, H, BT * RANK_AB), device=device, dtype=dtype, 
    )
    dA_qk = torch.zeros(
        size=(B, T, H, BT), device=device, dtype=dtype, 
    )
    dA_qk_ref = torch.zeros_like(dA_qk)
    dA_qb = torch.zeros(
        size=(B, T, H, BT * RANK_AB), device=device, dtype=dtype, 
    )
    dA_qb_ref = torch.zeros_like(dA_qb)
    dv_new = torch.zeros(
        size=(B, T * RANK_AB, H, V), device=device, dtype=dtype, 
    )
    dv_new_ref = torch.zeros_like(dv_new)

    chunk_dplr_bwd_kernel_dAu[grid](
        v, # [B, T, H, V]
        do, # [B, T, H, V]
        v_new, # [B, T * RANK_AB, H, V]
        A_qb, # [B, T, H, BT * RANK_AB]
        dA_qk, # [B, T, H, BT]
        dA_qb, # [B, T, H, BT * RANK_AB]
        dv_new, # [B, T * RANK_AB, H, V]
        scale=1.,
        T=T,
        RANK_AB=RANK_AB, 
        BT_AB=BT * RANK_AB, 
        H=H,
        V=V,
        BV=BV,
        BT=BT, 
    )

    o_i_kv = torch.arange(0, BT, device=device)
    o_i_ab = torch.repeat_interleave(o_i_kv, repeats=2, dim=0)
    
    mask_qk = (o_i_kv[:, None] >= o_i_kv[None, :])[None, :, None, :]
    mask_qb = (o_i_kv[:, None] >= o_i_ab[None, :])[None, :, None, :]

    for i_t in range(NT):
        dA_qk_ref[:, i_t * BT: (i_t + 1) * BT, :, :] = torch.einsum(
            'b m h k, b n h k -> b m h n', 
            do[:, i_t * BT: (i_t + 1) * BT, :, :], 
            v[:, i_t * BT: (i_t + 1) * BT, :, :], 
        ) * mask_qk
        dA_qb_ref[:, i_t * BT: (i_t + 1) * BT, :, :] = torch.einsum(
            'b m h k, b n h k -> b m h n', 
            do[:, i_t * BT: (i_t + 1) * BT, :, :], 
            v_new[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, :, :], 
        ) * mask_qb
        dv_new_ref[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, :, :] = torch.einsum(
            'b t h c, b t h v -> b c h v', 
            A_qb[:, i_t * BT: (i_t + 1) * BT, :, :], # [B, BT, H, BT * RANK_AB]
            do[:, i_t * BT: (i_t + 1) * BT, :, :], # [B, BT, H, V]
        ) # [BT_AB, BV]

    assert torch.allclose(dA_qb, dA_qb_ref, rtol=1e-3, atol=1e-3)
    assert torch.allclose(dA_qk, dA_qk_ref, rtol=1e-3, atol=1e-3)
    assert torch.allclose(dv_new, dv_new_ref, rtol=1e-3, atol=1e-3)