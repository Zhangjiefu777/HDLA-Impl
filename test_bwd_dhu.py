import triton
import torch
import torch.nn.functional as F

from HRDPLR.bwd_triton import chunk_dplr_bwd_kernel_dhu
from einops import einsum

if __name__ == '__main__':
    B = 4
    H = 4
    T = 1024
    K = 64
    V = 96

    BK = K
    NK = triton.cdiv(K, BK)

    BV = 32
    NV = triton.cdiv(V, BV)

    RANK_AB = 2

    BT = 32
    NT = triton.cdiv(T, BT)

    BC = 16
    
    device = torch.device('cuda:0')
    dtype = torch.float32

    q = torch.randn(
        size=(B, T, H, K), device=device, dtype=dtype, 
    )
    b = torch.randn(
        size=(B, T * RANK_AB, H, K), device=device, dtype=dtype, 
    )
    b /= b.abs().max()
    w = torch.randn(
        size=(B, T * RANK_AB, H, K), device=device, dtype=dtype, 
    ) 
    w /= w.abs().max()

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

    qg = (q * torch.exp(gk)).to(q.dtype)
    bg = (b * torch.repeat_interleave(
        torch.exp(gk), repeats=2, dim=1, 
    )).to(b.dtype)

    dht = torch.randn(
        size=(B, H, K, V), device=device, dtype=dtype, 
    )

    dh0 = torch.zeros_like(dht)
    dh0_ref = torch.zeros_like(dht)

    do = torch.randn(
        size=(B, T, H, V), device=device, dtype=dtype, 
    )

    dh = torch.zeros(
        size=(B, NT, H, K, V), device=device, dtype=dtype, 
    )
    dh_ref = torch.zeros_like(dh)

    dv = torch.randn(
        size=(B, T * RANK_AB, H, V), device=device, dtype=dtype, 
    )

    dv2 = torch.zeros_like(dv)
    dv2_ref = torch.zeros_like(dv2)

    grid = (NK, NV, B * H)
    chunk_dplr_bwd_kernel_dhu[grid](
        qg, # [B, T, H, K]
        bg, # [B, T * RANK_AB, H, K]
        w, # [B, T * RANK_AB, H, K]
        gk, # [B, T, H, K]
        dht, # [B, H, K, V]
        dh0, # [B, H, K, V]
        do, # [B, H, ]
        dh, # [B, NT, H, K, V]
        dv, # (dv_new_intra) [B, T * RANK_AB, H, V]
        dv2, # (dv_new) [B, T * RANK_AB, H, V]
        T,
        H,
        K,
        V,
        BT,
        BC,
        BK,
        BV,
        RANK_AB, 
    )

    dh_now = dht.clone()
    for i_t in range(NT - 1, -1 , -1):
        # store the gradient of H_{[t + 1]}, i.e. the hidden state after the i_t-th block ends
        dh_ref[:, i_t, :, :, :] = dh_now 
        # [B, T * RANK_AB, H, V]
        for i_k in range(NK):
            dv2_ref[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, :, :] = (
                dv[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, :, :] # [B, BT_AB, H, V]
                + ( 
                    bg[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, :, :].transpose(1, 2) # [B, H, BT_AB, K]
                    @ 
                    dh_now.to(bg.dtype) # [B, H, K, V]
                ).transpose(1, 2) # [B, H, BT_AB, V] -> [B, BT_AB, H, V]
            ) # [B, BT_AB, H, V]
        g_last = torch.exp(
            gk[:, (i_t + 1) * BT - 1, :, :]
        ) # [B, H, K]
        dh_now = (
            dh_now * g_last[..., None] 
            + torch.einsum(
                'b t h k, b t h v -> b h k v', 
                qg[:, i_t * BT: (i_t + 1) * BT, :, :], # [B, BT, H, K]
                do[:, i_t * BT: (i_t + 1) * BT, :, :], # [B, BT, H, V]
            )
            + torch.einsum(
                'b t h k, b t h v -> b h k v', 
                w[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, :, :], # [B, BT_AB, H, K]
                dv2_ref[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, :, :], # [B, BT_AB, H, V]
            )
        ).to(dtype)
    dh0_ref = dh_now

    assert torch.allclose(dh, dh_ref, rtol=5e-2, atol=5e-2)
    assert torch.allclose(dh0, dh0_ref, rtol=5e-2, atol=5e-2)
    assert torch.allclose(dv2, dv2_ref, rtol=5e-2, atol=5e-2)