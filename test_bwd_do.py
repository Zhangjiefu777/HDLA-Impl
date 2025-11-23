import triton
import torch
import torch.nn.functional as F

from HRDPLR.bwd_triton import chunk_dplr_bwd_o_kernel

if __name__ == '__main__':


    B = 2
    T = 256
    H = 4
    K = 96
    V = 64
    RANK_AB = 2
    
    BT = 32
    BK = 32
    BV = 16

    NT = triton.cdiv(T, BT)
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)

    device = torch.device('cuda')
    dtype = torch.float32

    v = torch.randn(
        size=(B, T, H, V), device=device, dtype=dtype
    )
    v_new = torch.randn(
        size=(B, T * RANK_AB, H, V), device=device, dtype=dtype
    )
    h = torch.randn(
        size=(B, NT, H, K, V), device=device, dtype=dtype
    )
    do = torch.randn(
        size=(B, T, H, V), device=device, dtype=dtype
    )
    dh = torch.randn(
        size=(B, NT, H, K, V), device=device, dtype=dtype
    )
    w = torch.randn(
        size=(B, T * RANK_AB, H, K), device=device, dtype=dtype
    )
    dv_new = torch.randn(
        size=(B, T * RANK_AB, H, V), device=device, dtype=dtype
    )

    g = F.logsigmoid(
        torch.randn(
            size=(B, T, H, K), device=device, dtype=torch.float32, 
        )
    )
    gk = torch.zeros_like(g)
    for i_t in range(NT):
        gk[:, i_t * BT: (i_t + 1) * BT, :, :] = torch.cumsum(
            g[:, i_t * BT: (i_t + 1) * BT, :, :], dim=1, 
        )
    k = torch.randn(
        size=(B, T, H, K), device=device, dtype=torch.float32, 
    )
    b = torch.randn(
        size=(B, T * RANK_AB, H, K), device=device, dtype=torch.float32, 
    )

    dk_triton = torch.zeros(
        size=(B, T, H, K), device=device, dtype=dtype
    )
    dk_ref = torch.zeros_like(dk_triton)

    db_triton = torch.zeros(
        size=(B, T * RANK_AB, H, K), device=device, dtype=dtype
    )
    db_ref = torch.zeros_like(db_triton)

    dq_triton = torch.zeros(
        size=(B, T, H, K), device=device, dtype=dtype
    )
    dq_ref = torch.zeros_like(dq_triton)

    dw_triton = torch.zeros(
        size=(B, T * RANK_AB, H, K), device=device, dtype=dtype
    )
    dw_ref = torch.zeros_like(dw_triton)

    dgk_last_triton = torch.zeros(
        size=(B, NT, H, K), device=device, dtype=dtype
    )
    dgk_last_ref = torch.zeros_like(dgk_last_triton)

    def grid(meta): return (NK, NT, B * H)
    chunk_dplr_bwd_o_kernel[grid](
        v, # [B, T, H, V]
        v_new, # [B, T * RANK_AB, H, V]
        h, # [B, NT, H, K, V]
        do, # [B, T, H, V]
        dh, # [B, NT, H, K, V]
        dk_triton,
        db_triton,
        w,
        dq_triton,
        dv_new, # (dv_new) [B, T * RANK_AB, H, K]
        dw_triton,
        gk,
        dgk_last_triton,
        k,
        b,
        None,
        None,
        T,
        H,
        K,
        V,
        RANK_AB, 
        BT,
        BT * RANK_AB, 
        BK, 
        BV,
    )

    for i_t in range(NT):
        dq_ref[:, i_t * BT: (i_t + 1) * BT, ...] = torch.einsum(
            'b c h v, b h k v -> b c h k', 
            do[:, i_t * BT: (i_t + 1) * BT, ...], # [B, BT, H, V]
            h[:, i_t, ...], # [B, H, K, V]
        )
        dk_ref[:, i_t * BT: (i_t + 1) * BT, ...] = torch.einsum(
            'b c h v, b h k v -> b c h k', 
            v[:, i_t * BT: (i_t + 1) * BT, ...], # [B, C, H, V]
            dh[:, i_t, ...], # [B, H, K, V]
        )
        db_ref[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...] = torch.einsum(
            'b c h v, b h k v -> b c h k', 
            v_new[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...], # [B, C * RANK_AB, H, V]
            dh[:, i_t, ...], # [B, H, K, V]
        )
        dw_ref[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...] = torch.einsum(
            'b c h v, b h k v -> b c h k', 
            dv_new[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...], # [B, C * RANK_AB, H, V]
            h[:, i_t, ...], # [B, H, K, V]
        )
        dgk_last_ref[:, i_t, ...] = torch.einsum(
            'b h k v, b h k v -> b h k', 
            h[:, i_t, ...], # [B, H, K, V]
            dh[:, i_t, ...], # [B, H, K, V]
        ) # [B, H, K]
        dgk_last_ref[:, i_t, ...] *= torch.exp(gk[:, (i_t + 1) * BT - 1, ...]) # [B, H, K]
        dgk_last_ref[:, i_t, ...] += torch.einsum(
            'b c h k, b c h k -> b h k', 
            b[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...], # [B, C * RANK_AB, H, K]
            db_ref[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...], # [B, C * RANK_AB, H, K]
        )
        dgk_last_ref[:, i_t, ...] += torch.einsum(
            'b c h k, b c h k -> b h k', 
            k[:, i_t * BT: (i_t + 1) * BT, ...], # [B, C, H, K]
            dk_ref[:, i_t * BT: (i_t + 1) * BT, ...], # [B, C, H, K]
        )

    assert torch.allclose(dq_ref, dq_triton, rtol=5e-2, atol=5e-2)
    assert torch.allclose(dk_ref, dk_triton, rtol=5e-2, atol=5e-2)
    assert torch.allclose(db_ref, db_triton, rtol=5e-2, atol=5e-2)
    assert torch.allclose(dw_ref, dw_triton, rtol=5e-2, atol=5e-2)
    assert torch.allclose(dgk_last_ref, dgk_last_triton, rtol=5e-2, atol=5e-2)