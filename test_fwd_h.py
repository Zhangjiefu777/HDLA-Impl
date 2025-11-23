import triton
import torch
import torch.nn.functional as F

from HRDPLR.fwd_triton import chunk_dplr_fwd_kernel_h

if __name__ == '__main__':

    B = 2
    H = 4
    T = 512
    K = 128
    V = 64
    RANK_AB = 2

    BT = 32
    BK = K
    BV = 16
    BC = BT
    
    NT = triton.cdiv(T, BT)
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)

    device = torch.device('cuda')
    dtype = torch.float32

    kg = torch.randn(size=(B, T, H, K), device=device, dtype=dtype)
    kg /= (torch.norm(kg, p=2, dim=-1, keepdim=True) + 1e-6)

    v = torch.randn(size=(B, T, H, V), device=device, dtype=dtype)

    w = torch.randn(size=(B, T * RANK_AB, H, K), device=device, dtype=dtype)
    w /= w.abs().max()
    
    bg = torch.randn(size=(B, T * RANK_AB, H, K), device=device, dtype=dtype)
    bg /= (torch.norm(bg, p=2, dim=-1, keepdim=True) + 1e-6)

    u = torch.randn(size=(B, T * RANK_AB, H, V), device=device, dtype=dtype)
    u /= u.abs().max()

    h0 = torch.randn(size=(B, H, K, V), device=device, dtype=dtype)

    h0 /= h0.abs().max()
    
    v_new_triton = torch.zeros(size=(B, T * RANK_AB, H, V), device=device, dtype=dtype)
    h_triton = torch.zeros(size=(B, NT, H, K, V), device=device, dtype=dtype)
    ht_triton = torch.zeros(size=(B, H, K, V), device=device, dtype=dtype)

    v_new_ref = torch.zeros_like(v_new_triton)
    h_ref = torch.zeros_like(h_triton)
    ht_ref = torch.zeros_like(ht_triton)

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

    for i_t in range(NT):
        gk_last = gk[:, (i_t + 1) * BT - 1: (i_t + 1) * BT, ...] # [B, 1, H, K]
        gk_tilde = gk_last - gk[:, i_t * BT: (i_t + 1) * BT, ...] # [B, BT, H, K]
        gk_tilde_ab = torch.repeat_interleave(
            gk_tilde, repeats=RANK_AB, dim=1, 
        )
        kg[:, i_t * BT: (i_t + 1) * BT, ...] *= torch.exp(gk_tilde)
        bg[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB] *= torch.exp(gk_tilde_ab)

    def grid(meta): return [NK, NV, B * H]
    chunk_dplr_fwd_kernel_h[grid](
        kg, # [B, T, H, K]
        v, # [B, H, T, V]        
        w, # [B, H, T * RANK_AB, K]
        bg, # [B, H, T * RANK_AB, K]
        u, # [B, H, T * RANK_AB, K]
        v_new_triton, # [B, H, T * RANK_AB, V]
        gk, # [B, H, T, K]
        h_triton, # [B, H, NT, K, V]
        h0, # [B, H, K, V]
        ht_triton, # [B, H, K, V]
        T,
        H,
        K,
        V,
        BT,
        BC,
        RANK_AB, 
        BC * RANK_AB, 
        BK,
        BV,
        NT,
    )

    h_ref[:, 0, ...] = h0
    for i_t in range(NT):
        gk_last = gk[:, (i_t + 1) * BT - 1, ...] # [B, H, K]
        v_new_ref[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...] = (
            torch.einsum(
                'b c h k, b h k v -> b c h v', # A @ B 
                w[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...], 
                h_ref[:, i_t, ...]
            ) + u[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...] 
        )
        # [B, H, K, V]
        if i_t < NT - 1:
            h_ref[:, i_t + 1, ...] = (
                (
                    torch.exp(gk_last)[..., None] * h_ref[:, i_t, ...]
                ) + torch.einsum(
                    'b c h k, b c h v -> b h k v', # A.T @ B 
                    kg[:, i_t * BT: (i_t + 1) * BT, ...], # [B, BT, H, K]
                    v[:, i_t * BT: (i_t + 1) * BT, ...], # [B, BT, H, V]
                ) + torch.einsum(
                    'b c h k, b c h v -> b h k v', # A.T @ B 
                    bg[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...], # [B, BT * RANK_AB, H, K]
                    v_new_ref[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...], # [B, BT * RANK_AB, H, V]
                )
            )
        else:
            ht_ref = (
                (
                    torch.exp(gk_last)[..., None] * h_ref[:, i_t, ...]
                ) + torch.einsum(
                    'b c h k, b c h v -> b h k v', # A.T @ B 
                    kg[:, i_t * BT: (i_t + 1) * BT, ...], # [B, BT, H, K]
                    v[:, i_t * BT: (i_t + 1) * BT, ...], # [B, BT, H, V]
                ) + torch.einsum(
                    'b c h k, b c h v -> b h k v', # A.T @ B 
                    bg[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...], # [B, BT * RANK_AB, H, K]
                    v_new_ref[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...], # [B, BT * RANK_AB, H, V]
                )
            )

    
    
    assert torch.allclose(v_new_ref, v_new_triton, rtol=5e-2, atol=5e-2)
    assert torch.allclose(h_ref, h_triton, rtol=5e-2, atol=5e-2)
    assert torch.allclose(ht_ref, ht_triton, rtol=5e-2, atol=5e-2)

    print("Test passed!")