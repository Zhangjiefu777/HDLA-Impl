import triton
import torch
import torch.nn.functional as F

from HRDPLR.bwd_triton import chunk_hrdplr_bwd_kernel_intra
from einops import rearrange

if __name__ == '__main__':
    B = 2
    T = 256
    H = 4
    K = 96
    V = 64
    RANK_AB = 2
    
    BT = 32
    BC = BT
    BK = 32
    BV = 16

    NT = triton.cdiv(T, BT)
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)

    device = torch.device('cuda')
    dtype = torch.float32

    q = torch.randn(size=(B, T, H, K), device=device, dtype=dtype)
    k = torch.randn(size=(B, T, H, K), device=device, dtype=dtype)
    a = torch.randn(size=(B, T * RANK_AB, H, K), device=device, dtype=dtype)
    b = torch.randn(size=(B, T * RANK_AB, H, K), device=device, dtype=dtype)

    g = F.logsigmoid(
        torch.randn(
            size=(B, T, H, K), device=device, dtype=torch.float32, 
        )
    )
    gi = torch.zeros_like(g)
    for i_t in range(NT):
        gi[:, i_t * BT: (i_t + 1) * BT, :, :] = torch.cumsum(
            g[:, i_t * BT: (i_t + 1) * BT, :, :], dim=1, 
        )
    ge = gi - g
    gi_ab = torch.repeat_interleave(gi, repeats=RANK_AB, dim=1)
    ge_ab = torch.repeat_interleave(ge, repeats=RANK_AB, dim=1)

    dAqk = torch.randn(size=(B, T, H, BT), device=device, dtype=dtype)
    dAqb = torch.randn(size=(B, T, H, BT * RANK_AB), device=device, dtype=dtype)
    dAak = torch.randn(size=(B, T * RANK_AB, H, BT), device=device, dtype=dtype)
    dAab = torch.randn(size=(B, T * RANK_AB, H, BT * RANK_AB), device=device, dtype=dtype)

    dqg = torch.randn_like(q)
    dkg = torch.randn_like(k)
    dag = torch.randn_like(a)
    dbg = torch.randn_like(b)

    dq_triton = torch.zeros_like(q)
    dq_ref = torch.zeros_like(q)

    dk_triton = torch.zeros_like(k)
    dk_ref = torch.zeros_like(k)

    da_triton = torch.zeros_like(a)
    da_ref = torch.zeros_like(a)

    db_triton = torch.zeros_like(b)
    db_ref = torch.zeros_like(b)

    dgk_triton = torch.zeros_like(gi)
    dgk_ref = torch.zeros_like(gi)

    dgk_offset_triton = torch.zeros_like(gi)
    dgk_offset_ref = torch.zeros_like(gi)

    scale = 1

    grid = [NK, NT, B * H]
    chunk_hrdplr_bwd_kernel_intra[grid](
        q, 
        k, 
        a, 
        b, 
        gi, 
        ge, 
        dAqk, 
        dAqb, 
        dAak, 
        dAab,
        dq_triton,
        dk_triton,
        da_triton,
        db_triton,
        dqg,
        dkg,
        dag,
        dbg,
        dgk_triton,
        dgk_offset_triton,
        None,
        None,
        scale,
        T,
        RANK_AB, 
        H,
        K,
        BT,
        BT * RANK_AB, 
        BC,
        BC * RANK_AB, 
        BK=BK,
        IS_VARLEN=False,
        GATHER_SUPPORTED=False, 
    )

    range_BT = torch.arange(0, BT, device=device) # [BT, ]
    range_BT_AB = torch.repeat_interleave(range_BT, repeats=RANK_AB, dim=0) # [BT * RANK_AB]

    for i_t in range(NT):
        
        dAqk_block = dAqk[:, i_t * BT: (i_t + 1) * BT, ...] # [B, BT, H, BT]
        dAqb_block = dAqb[:, i_t * BT: (i_t + 1) * BT, ...] # [B, BT, H, BT * RANK_AB]
        dAak_block = dAak[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...] # [B, BT * RANK_AB, H, BT]
        dAab_block = dAab[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...] # [B, BT * RANK_AB, H, BT]

        gi_block = gi[:, i_t * BT: (i_t + 1) * BT, ...]
        ge_block = ge[:, i_t * BT: (i_t + 1) * BT, ...]
        gi_ab_block = gi_ab[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...]
        ge_ab_block = ge_ab[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...]
        
        for j in range(BT):
            
            dAqk_row_j = dAqk_block[:, j: j + 1, ...] # [B, 1, H, BT]
            dAqb_row_j = dAqb_block[:, j: j + 1, ...] # [B, 1, H, BT * RANK_AB]
            dAqk_column_j = dAqk_block[..., j: j + 1] # [B, BT, H, 1]
            dAak_column_j = dAak_block[..., j: j + 1] # [B, BT * RANK_AB, H, 1]

            q_j = q[:, i_t * BT + j: i_t * BT + j + 1, ...] # [B, 1, H, BT]
            k_j = k[:, i_t * BT + j: i_t * BT + j + 1, ...] # [B, 1, H, BT]
            gi_j = gi[:, i_t * BT + j: i_t * BT + j + 1, ...] # [B, 1, H, BT]
            ge_j = ge[:, i_t * BT + j: i_t * BT + j + 1, ...] # [B, 1, H, BT]

            mask_q = (range_BT >= j) # q index >= k/b index, [BT, ]
            mask_a = (range_BT_AB > j) # a index > k/b index, [BT * RANK_AB]

            mask_k_i = (range_BT <= j) # masking the gradient from q
            mask_k_e = (range_BT < j) # masking the gradient from b

            mask_b_i = (range_BT_AB <= j) # masking the gradient from q
            mask_b_e = (range_BT_AB < j) # masking the gradient from b

            dq_ref[:, i_t * BT: (i_t + 1) * BT, ...] += torch.einsum(
                'b r h m, b m h c -> b r h c', 
                dAqk_column_j, # [B, BT, H, 1]
                k_j, # [B, 1, H, K]
            ) * torch.exp(gi[:, i_t * BT: (i_t + 1) * BT, ...] - gi_j) * mask_q[None, :, None, None]

            da_ref[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...] += torch.einsum(
                'b r h m, b m h c -> b r h c', 
                dAak_column_j, # [B, BT * RANK_AB, H, 1]
                k_j, # [B, 1, H, K]
            ) * torch.exp(ge_ab[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...] - gi_j) * mask_a[None, :, None, None]

            dk_ref[:, i_t * BT: (i_t + 1) * BT, ...] += torch.einsum(
                'b m h r, b m h c -> b r h c', # A.T @ B
                dAqk_row_j, # [B, 1, H, BT]
                q_j, # [B, 1, H, K]
            ) * torch.exp(gi_j - gi[:, i_t * BT: (i_t + 1) * BT, ...]) * mask_k_i[None, :, None, None]

            db_ref[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...] += torch.einsum(
                'b m h r, b m h c -> b r h c', # A.T @ B
                dAqb_row_j, # [B, 1, H, BT * RANK_AB]
                q_j, # [B, 1, H, K]
            ) * torch.exp(gi_j - gi_ab[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...]) * mask_b_i[None, :, None, None]

            for r in range(RANK_AB):
                dAak_row_jr = dAak_block[:, (j * RANK_AB + r): (j * RANK_AB + r) + 1, ...] # [B, 1, H, BT]
                dAab_row_jr = dAab_block[:, (j * RANK_AB + r): (j * RANK_AB + r) + 1, ...] # [B, 1, H, BT]
                dAqb_column_jr = dAqb_block[..., (j * RANK_AB + r): (j * RANK_AB + r) + 1] # [B, BT, H, 1]
                dAab_column_jr = dAab_block[..., (j * RANK_AB + r): (j * RANK_AB + r) + 1] # [B, BT * RANK_AB, H, 1]
        
                a_jr = a[:, i_t * BT * RANK_AB + j * RANK_AB + r: i_t * BT * RANK_AB + j * RANK_AB + r + 1, ...]
                b_jr = b[:, i_t * BT * RANK_AB + j * RANK_AB + r: i_t * BT * RANK_AB + j * RANK_AB + r + 1, ...]

                dq_ref[:, i_t * BT: (i_t + 1) * BT, ...] += torch.einsum(
                    'b r h m, b m h c -> b r h c', 
                    dAqb_column_jr, # [B, BT, H, 1]
                    b_jr, # [B, 1, H, K]
                ) * torch.exp(gi[:, i_t * BT: (i_t + 1) * BT, ...] - gi_j) * mask_q[None, :, None, None]

                da_ref[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...] += torch.einsum(
                    'b r h m, b m h c -> b r h c', 
                    dAab_column_jr, # [B, BT * RANK_AB, H, 1]
                    b_jr, # [B, 1, H, K]
                ) * torch.exp(ge_ab[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...] - gi_j) * mask_a[None, :, None, None]

                dk_ref[:, i_t * BT: (i_t + 1) * BT, ...] += torch.einsum(
                    'b m h r, b m h c -> b r h c', # A.T @ B 
                    dAak_row_jr, # [B, 1, H, BT]
                    a_jr, # [B, 1, H, K]
                ) * torch.exp(ge_j - gi[:, i_t * BT: (i_t + 1) * BT, ...]) * mask_k_e[None, :, None, None]

                db_ref[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...] += torch.einsum(
                    'b m h r, b m h c -> b r h c', 
                    dAab_row_jr, # [B, 1, H, BT * RANK_AB]
                    a_jr, # [B, 1, H, K]
                ) * torch.exp(ge_j - gi_ab[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...]) * mask_b_e[None, :, None, None]

        dq_ref[:, i_t * BT: (i_t + 1) * BT, ...] += (
            dqg[:, i_t * BT: (i_t + 1) * BT, ...] * torch.exp(gi[:, i_t * BT: (i_t + 1) * BT, ...])
        )
        da_ref[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...] += (
            dag[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...] * torch.exp(ge_ab[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...])
        )

        gi_n = gi[:, (i_t + 1) * BT - 1: (i_t + 1) * BT, ...] # [B, 1, H, K]
        tmp_k = torch.exp(gi_n - gi[:, i_t * BT: (i_t + 1) * BT, ...]) # [B, BT, H, K]
        tmp_b = torch.exp(gi_n - gi_ab[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...]) # [B, BT * RANK_AB, H, K]

        dk_ref[:, i_t * BT: (i_t + 1) * BT, ...] += (
            dkg[:, i_t * BT: (i_t + 1) * BT, ...] * tmp_k
        )
        db_ref[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...] += (
            dbg[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...] * tmp_b
        )

        dgk_ref[:, i_t * BT: (i_t + 1) * BT, ...] = (
            dq_ref[:, i_t * BT: (i_t + 1) * BT, ...] * q[:, i_t * BT: (i_t + 1) * BT, ...]
            + torch.einsum(
                'b c r h k, b c r h k -> b c h k', 
                rearrange(
                    da_ref[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...], 
                    'b (c r) h k -> b c r h k', r=RANK_AB, 
                ), 
                rearrange(
                    a[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...], 
                    'b (c r) h k -> b c r h k', r=RANK_AB, 
                ), # [B, BT * RANK_AB, H, K]
            )
            - dk_ref[:, i_t * BT: (i_t + 1) * BT, ...] * k[:, i_t * BT: (i_t + 1) * BT, ...]
            - torch.einsum(
                'b c r h k, b c r h k -> b c h k', 
                rearrange(
                    db_ref[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...], 
                    'b (c r) h k -> b c r h k', r=RANK_AB, 
                ), 
                rearrange(
                    b[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...], 
                    'b (c r) h k -> b c r h k', r=RANK_AB, 
                ), # [B, BT * RANK_AB, H, K]
            )
        )
        dgk_offset_ref[:, i_t * BT: (i_t + 1) * BT, ...] = torch.einsum(
            'b c r h k, b c r h k -> b c h k', 
            rearrange(
                da_ref[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...], 
                'b (c r) h k -> b c r h k', r=RANK_AB, 
            ), # [B, BT * RANK_AB, H, K]
            rearrange(
                a[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...], 
                'b (c r) h k -> b c r h k', r=RANK_AB, 
            ), # [B, BT * RANK_AB, H, K]
        )

    assert torch.allclose(dq_triton, dq_ref, rtol=5e-2, atol=5e-2)
    assert torch.allclose(dk_triton, dk_ref, rtol=5e-2, atol=5e-2)
    assert torch.allclose(da_triton, da_ref, rtol=5e-2, atol=5e-2)
    assert torch.allclose(db_triton, db_ref, rtol=5e-2, atol=5e-2)
    assert torch.allclose(dgk_triton, dgk_ref, rtol=5e-2, atol=5e-2)
    assert torch.allclose(dgk_offset_triton, dgk_offset_ref, rtol=5e-2, atol=5e-2)