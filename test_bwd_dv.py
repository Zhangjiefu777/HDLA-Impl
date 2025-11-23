import triton
import torch
import torch.nn.functional as F

from HRDPLR.bwd_triton import chunk_dplr_bwd_kernel_dv

if __name__ == '__main__':

    dtype = torch.float32
    device = torch.device('cuda')

    B = 2
    T = 256
    H = 4
    BT = 32
    NT = triton.cdiv(T, BT)

    K = 96
    V = 64

    BK = 32
    BV = 32

    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)

    A_qk = torch.randn(
        size=(B, T, H, BT), device=device, dtype=dtype, 
    )
    kg = torch.randn(
        size=(B, T, H, K), device=device, dtype=dtype, 
    )
    do = torch.randn(
        size=(B, T, H, V), device=device, dtype=dtype, 
    )
    dh = torch.randn(
        size=(B, NT, H, K, V), device=device, dtype=dtype, 
    )

    dv_triton = torch.zeros(
        size=(B, T, H, V), device=device, dtype=dtype, 
    )
    dv_ref = torch.zeros_like(dv_triton)

    range_BT = torch.arange(0, BT, device=device)
    tril_mask = (range_BT[:, None] >= range_BT[None, :])

    for i_t in range(NT):
        A_qk[:, i_t * BT: (i_t + 1) * BT, ...] *= tril_mask[None, :, None, :]


    def grid(meta): return [triton.cdiv(V, meta['BV']), NT, B * H]
    chunk_dplr_bwd_kernel_dv[grid](
        A_qk, # [B, T, H, BT]
        kg, # [B, T, H, K]
        do, # [B, T, H, V]
        dv_triton, # [B, T, H, V]
        dh, # [B, NT, H, K, V]
        T,
        H,
        K,
        V,
        BT,
    )

    for i_t in range(NT):
        dv_ref[:, i_t * BT: (i_t + 1) * BT, :, :] = (
            torch.einsum(
                'b c h k, b h k v -> b c h v', 
                kg[:, i_t * BT: (i_t + 1) * BT, :, :], # [B, C, H, K]
                dh[:, i_t, ...], # [B, H, K, V]
            ) + torch.einsum(
                'b c h t, b c h v -> b t h v', 
                A_qk[:, i_t * BT: (i_t + 1) * BT, :, :], # [B, C, H, BT]
                do[:, i_t * BT: (i_t + 1) * BT, :, :], # [B, C, H, V]
            )
        )
    
    assert torch.allclose(dv_triton, dv_ref, atol=5e-2, rtol=5e-2)
    print("Test passed!")