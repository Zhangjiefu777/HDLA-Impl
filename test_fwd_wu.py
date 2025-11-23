import triton
import torch
import torch.nn.functional as F

from HRDPLR.fwd_triton import fwd_wu_kernel

if __name__ == '__main__':

    B = 2
    H = 4
    T = 256
    K = 128
    V = 64
    RANK_AB = 2

    BT = 16
    BK = K
    BV = 16
    BC = BT
    
    NT = triton.cdiv(T, BT)
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)

    device = torch.device('cuda')
    dtype = torch.float32

    ag = torch.randn(size=(B, T * RANK_AB, H, K), device=device, dtype=dtype)
    v = torch.randn(size=(B, T, H, V), device=device, dtype=dtype)
    A_ab_inv = torch.randn(size=(B, T * RANK_AB, H, BT * RANK_AB), device=device, dtype=dtype)
    A_ak = torch.randn(size=(B, T * RANK_AB, H, BT), device=device, dtype=dtype)

    range_BT = torch.arange(0, BT, device=device, dtype=dtype)
    range_BT_AB = torch.repeat_interleave(range_BT, repeats=RANK_AB, dim=0)

    o_BT_AB = torch.arange(0, BT * RANK_AB, device=device, dtype=dtype)

    mask_Aab = o_BT_AB[:, None] >= o_BT_AB[None, :] # element-wise mask
    mask_Aak = range_BT_AB[:, None] > range_BT[None, :] # block-wise mask

    u_triton = torch.zeros(size=(B, T * RANK_AB, H, V), device=device, dtype=dtype)
    u_ref = torch.zeros_like(u_triton)
    w_triton = torch.zeros(size=(B, T * RANK_AB, H, K), device=device, dtype=dtype)
    w_ref = torch.zeros_like(w_triton)

    for i_t in range(NT):
        A_ab_inv[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...] *= mask_Aab[None, :, None, :]
        A_ak[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...] *= mask_Aak[None, :, None, :]
        u_ref[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...] = torch.einsum(
            'b r h m, b m h c -> b r h c', 
            torch.einsum(
                'b r h m, b m h c -> b r h c', 
                A_ab_inv[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...], 
                A_ak[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...], 
            ), 
            v[:, i_t * BT: (i_t + 1) * BT, ...]
        )
        w_ref[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...] = torch.einsum(
            'b r h m, b m h c -> b r h c', 
            A_ab_inv[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...], 
            ag[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...], 
        )

    grid = [NT, B * H]
    fwd_wu_kernel[grid](
        u_triton, # [B, T * RANK_AB, H, K]
        w_triton, # [B, T * RANK_AB, H, V]
        ag, # [B, T, H, K]
        v, # [B, T, H, V]
        A_ab_inv, # [B, T * RANK_AB, H, BT * RANK_AB]
        A_ak, # [B, T * RANK_AB, H, BT]
        T,
        H,
        K,
        V,
        BT,
        RANK_AB, 
        BT * RANK_AB, 
        BK,
        BV,
    )

    assert torch.allclose(u_ref, u_triton, rtol=5e-2, atol=5e-2)
    assert torch.allclose(w_ref, w_triton, rtol=5e-2, atol=5e-2)