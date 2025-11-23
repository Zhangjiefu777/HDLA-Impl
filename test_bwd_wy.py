import triton
import torch
import torch.nn.functional as F

from HRDPLR.bwd_triton import prepare_wy_repr_bwd_kernel

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

    A_ab_inv = torch.randn(
        size=(B, T * RANK_AB, H, BT * RANK_AB), device=device, dtype=dtype, 
    )
    A_ab_inv /= A_ab_inv.abs().max()

    A_ak = torch.randn(
        size=(B, T * RANK_AB, H, BT), device=device, dtype=dtype, 
    )
    A_ak /= A_ak.abs().max()

    ag = torch.randn(
        size=(B, T * RANK_AB, H, K), device=device, dtype=dtype, 
    )
    v = torch.randn(
        size=(B, T, H, V), device=device, dtype=dtype, 
    )
    dv0 = torch.randn_like(v)
    dw = torch.randn(
        size=(B, T * RANK_AB, H, K), device=device, dtype=dtype, 
    )
    du = torch.randn(
        size=(B, T * RANK_AB, H, V), device=device, dtype=dtype, 
    )

    range_BT = torch.arange(0, BT, device=device)
    range_BT_AB = torch.repeat_interleave(input=range_BT, repeats=RANK_AB, dim=0)
    mask_A_ab_inv = range_BT_AB[:, None] > range_BT_AB[None, :]
    mask_A_ak = range_BT_AB[:, None] > range_BT[None, :]

    for i_t in range(NT):
        A_ab_inv[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...] *= mask_A_ab_inv[None, :, None, :]
        A_ak[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...] *= mask_A_ak[None, :, None, :]

    dv_triton = torch.zeros_like(v)
    dv_ref = torch.zeros_like(dv_triton)

    dag_triton = torch.zeros(
        size=(B, T * RANK_AB, H, K), device=device, dtype=dtype, 
    )
    dag_ref = torch.zeros_like(dag_triton)

    dAak_triton = torch.zeros_like(A_ak)
    dAak_ref = torch.zeros_like(dAak_triton)

    dAab_triton = torch.zeros_like(A_ab_inv)
    dAab_ref = torch.zeros_like(dAab_triton)

    grid = [NT, B * H]
    prepare_wy_repr_bwd_kernel[grid](
        A_ab_inv,
        A_ak,
        ag,
        v,
        dw,
        du,
        dv_triton,
        dv0,
        dag_triton,
        dAak_triton,
        dAab_triton,
        None,
        None,
        T,
        RANK_AB, 
        H,
        K,
        V,
        BT,
        BT * RANK_AB, 
        BK,
        BV,
    )

    for i_t in range(NT):
        dv_ref[:, i_t * BT: (i_t + 1) * BT, ...] = (
            dv0[:, i_t * BT: (i_t + 1) * BT, ...]
            + 
            torch.einsum(
                'b c h t, b t h v -> b c h v', # c: BT; t: BT * RANK_AB
                torch.einsum(
                    'b t h c, b l h t -> b c h l', # c: BT; t: row dim of A_ak & column dim of A_ab_inv; l: row dim of A_ab_inv
                    A_ak[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...], # [B, BT_AB, H, BT]
                    A_ab_inv[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...], # [B, BT_AB, H, BT_AB]
                ), # [B, BT, H, BT_AB]
                du[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...], # [B, BT_AB, H, V]
            ) # [B, BT, H, V]
        )
        dag_ref[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...] = torch.einsum(
            'b l h t, b l h k -> b t h k', 
            A_ab_inv[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...], # [B, BT_AB, H, BT_AB]
            dw[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...], # [B, BT_AB, H, K]
        ) # [B, BT_AB, H, K]
        # dprod: the gradient of (A_ab_inv @ A_ak), which is also lower-triangular
        dprod = torch.einsum(
            'b t h v, b c h v -> b t h c', # t: BT * RANK_AB, c: BT
            du[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...], # [B, BT * RANK_AB, H, V]
            v[:, i_t * BT: (i_t + 1) * BT, ...], # [B, BT, H, V]
        ) * mask_A_ak[None, :, None, :] # [B, BT_AB, H, BT]
        dAak_ref[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...] = torch.einsum(
            'b t h r, b t h c -> b r h c', # t: row dim of dprod, c: column dim of dprod
            A_ab_inv[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...], # [B, BT * RANK_AB, H, BT * RANK_AB]
            dprod, # [B, BT * RANK_AB, H, BT]
        ) * mask_A_ak[None, :, None, :]
        dAab_inv_from_w = torch.einsum(
            'b r h v, b c h v -> b r h c', # r/c: the row/column dim of the resulting matrix
            dw[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...], 
            ag[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...], # transposed 
        ) * mask_A_ab_inv[None, :, None, :]
        dAab_inv_from_u = torch.einsum(
            'b r h m, b c h m -> b r h c', 
            dprod, # [B, BT * RANK_AB, H, BT]
            A_ak[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...], # transposed
        ) * mask_A_ab_inv[None, :, None, :]
        dAab_inv = dAab_inv_from_w + dAab_inv_from_u
        dAab_ref[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...] = torch.einsum(
            'b r h m, b c h m -> b r h c', # A @ B.T
            torch.einsum(
                'b m h r, b m h c -> b r h c', # A.T @ B
                A_ab_inv[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...], 
                dAab_inv, 
            ), 
            A_ab_inv[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...], 
        ) * mask_A_ab_inv[None, :, None, :]
        

    assert torch.allclose(dv_triton, dv_ref, rtol=5e-2, atol=5e-2)
    assert torch.allclose(dag_triton, dag_ref, rtol=5e-2, atol=5e-2)
    assert torch.allclose(dAab_triton, dAab_ref, rtol=5e-2, atol=5e-2)
    assert torch.allclose(dAak_triton, dAak_ref, rtol=5e-2, atol=5e-2)
    print("Test passed!")