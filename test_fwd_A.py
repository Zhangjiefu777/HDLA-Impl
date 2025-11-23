import triton
import torch
import torch.nn.functional as F

from HRDPLR.fwd_triton import (
    chunk_dplr_fwd_A_kernel_intra_sub_intra_rab_generalized, 
    chunk_dplr_fwd_A_kernel_intra_sub_inter_rab_generalized, 
)

def compute_A_ref_torch(
    operand_a_in: torch.Tensor, # [B, T * RANK_A, H, D]
    operand_b_in: torch.Tensor, # [B, T * RANK_B, H, D]
    g_in: torch.Tensor, 
    gi_in: torch.Tensor, 
    # g: 形状与operand_a相同
    rank_a: int, 
    rank_b: int, 
    BT: int, 
    USE_A: bool, 
):
    """
    The shape of g is the same as operand_a.
    That is: If operand_a is a ([B, T * RANK_AB, H, K]), then g should be gi_ab/ge_ab.
    """
    operand_a = operand_a_in.transpose(1, 2)
    operand_b = operand_b_in.transpose(1, 2)

    g = g_in.transpose(1, 2)
    gi = gi_in.transpose(1, 2)

    B, H, T_a, D = operand_a.shape

    T = T_a // rank_a
    A = torch.empty(
        size=(B, H, T * rank_a, BT * rank_b), dtype=operand_a.dtype, device=operand_a.device, 
    )
    
    NT = T // BT
    range_BT = torch.arange(0, BT, device=operand_a.device) # [BT, ]
    range_a = torch.repeat_interleave(
        input=range_BT, repeats=rank_a, dim=0, 
    ) # [BT * rank_a, ]
    for i_t in range(NT):
        
        for j in range(BT):
            if USE_A:
                condition = (range_a > j)[None, None, :, None]
            else:
                condition = (range_a >= j)[None, None, :, None]
            # A: [B, H, T * rank_a, BT * rank_b]
            A[:, :, i_t * (BT * rank_a): (i_t + 1) * (BT * rank_a), j * rank_b: (j + 1) * rank_b] = (
                operand_a[:, :, i_t * (BT * rank_a): (i_t + 1) * (BT * rank_a), :] * torch.where(
                    condition=condition, 
                    input=torch.exp(
                        g[:, :, i_t * (BT * rank_a): (i_t + 1) * (BT * rank_a), :] - gi[:, :, (i_t * BT + j) * rank_a: (i_t * BT + j) * rank_a + 1, :]
                    ), 
                    other=0.
                )
            ).to(operand_a.dtype) @ operand_b[:, :, (i_t * BT + j) * rank_b: (i_t * BT + j + 1) * rank_b, :].transpose(-2, -1)
            # [B, H, BT * rank_a, D] @ [B, H, D, rank_b] -> [B, H, BT * rank_a, rank_b]
        
    return A.transpose_(1, 2)

if __name__ == '__main__':
    B = 2
    H = 2
    T = 256
    K = 128
    V = 64

    BT = 64
    NT = triton.cdiv(T, BT)
    
    BC = 16
    NC = triton.cdiv(BT, BC)

    BK = K

    device = torch.device('cuda:0')
    dtype = torch.float32

    scale = 1
    RANK_KV = 1
    RANK_AB = 2

    q = torch.randn(
        size=(B, T, H, K), device=device, dtype=dtype, 
    )
    k = torch.randn(
        size=(B, T, H, K), device=device, dtype=dtype, 
    )
    v = torch.randn(
        size=(B, T, H, K), device=device, dtype=dtype, 
    )
    a = torch.randn(
        size=(B, T * RANK_AB, H, K), device=device, dtype=dtype, 
    )
    b = torch.randn(
        size=(B, T * RANK_AB, H, K), device=device, dtype=dtype, 
    )

    decay = torch.sigmoid(
        torch.randn(
            (B, T, H, K), device=device, dtype=torch.float32, 
        )
    )
    decay_log = torch.log(decay) # [B, T, H, K]

    decay_log_chunkwise_cumsum = torch.zeros_like(decay_log)
    for i_t in range(NT):
        decay_log_chunkwise_cumsum[:, i_t * BT: (i_t + 1) * BT, :, :] = torch.cumsum(
            input=decay_log[:, i_t * BT: (i_t + 1) * BT, :, :],
            dim=1, 
        )
    gi_kv = decay_log_chunkwise_cumsum
    ge_kv = decay_log_chunkwise_cumsum - decay_log

    Aqk = torch.zeros(
        size=(B, T, H, BT), device=device, dtype=dtype, 
    )
    Aqb = torch.zeros(
        size=(B, T, H, BT * RANK_AB), device=device, dtype=dtype, 
    )
    Aak = torch.zeros(
        size=(B, T * RANK_AB, H, BT), device=device, dtype=dtype, 
    )
    Aab = torch.zeros(
        size=(B, T * RANK_AB, H, BT * RANK_AB), device=device, dtype=dtype, 
    )

    Aqk_ref_torch = torch.empty_like(Aqk)
    Aqb_ref_torch = torch.empty_like(Aqb)
    Aak_ref_torch = torch.empty_like(Aak)
    Aab_ref_torch = torch.empty_like(Aab)

    qg = torch.empty_like(q)
    kg = torch.empty_like(k)
    ag = torch.empty_like(a)
    bg = torch.empty_like(b)

    qg_ref = torch.empty_like(q)
    kg_ref = torch.empty_like(k)
    ag_ref = torch.empty_like(a)
    bg_ref = torch.empty_like(b)

    gi_ab = torch.repeat_interleave(gi_kv, repeats=RANK_AB, dim=1)
    ge_ab = torch.repeat_interleave(ge_kv, repeats=RANK_AB, dim=1)

    grid = (NT, NC, B * H)
    chunk_dplr_fwd_A_kernel_intra_sub_intra_rab_generalized[grid](
        q=q, # [B, T, H, K] / [B, T, H, K]
        k=k, # [B, T, H, K] / [B, T, H, K]
        a=a, # [B, T * RANK_AB, H, K] / [B, T * RANK_AB, H, K]
        b=b, # [B, T * RANK_AB, H, K] / [B, T * RANK_AB, H, K]
        gi=gi_kv, # [B, T, H, K]
        ge=ge_kv, # [B, T, H, K]
        # the following 4 tensors are initialized as empty
        qg=qg, # [B, T, H, K]
        kg=kg, # [B, T, H, K]
        ag=ag, # [B, T * RANK_AB, H, K]
        bg=bg, # [B, T * RANK_AB, H, K] 
        # intermediate results
        Aqk=Aqk, # [B, T, H, BT]
        Aqb=Aqb, # [B, T * RANK_AB, H, BT * RANK_AB]
        Aab=Aab, # [B, T * RANK_AB, H, BT * RANK_AB]
        Aak=Aak, # [B, T * RANK_AB, H, BT]
        RANK_AB=RANK_AB, 
        scale=scale,
        T=T,
        H=H,
        K=K,
        BT=BT, # BT: coarse-grained chunk size along T dim
        BC=BC, # BC: fine-grained chunk size along T dim
        BC_AB=BC * RANK_AB, # BC * RANK_AB
        BK=BK,
        NC=NC,
    )

    grid = (NT, NC * NC, B * H)
    chunk_dplr_fwd_A_kernel_intra_sub_inter_rab_generalized[grid](
        q, # [B, T, H, K]
        k, # [B, T, H, K]
        a, # [B, T, H, K]
        b, # [B, T, H, K]
        gi_kv, # [B, T, H, K]
        ge_kv, # [B, T, H, K]
        # intermediate results
        Aqk, # [B, T, H, BT]
        Aqb, # [B, T, H, BT * RANK_AB]
        Aab, # [B, T * RANK_AB, H, BT * RANK_AB]
        Aak, # [B, T * RANK_AB, H, BT]
        RANK_AB, 
        scale, 
        T,
        H,
        K,
        BT,
        BC,
        BC * RANK_AB, 
        NC=NC,
    )

    Aqk_ref_torch = compute_A_ref_torch(
        operand_a_in=q, operand_b_in=k, g_in=gi_kv, gi_in=gi_kv, rank_a=RANK_KV, rank_b=RANK_KV, BT=BT, USE_A=False, 
    )
    Aqb_ref_torch = compute_A_ref_torch(
        operand_a_in=q, operand_b_in=b, g_in=gi_kv, gi_in=gi_kv, rank_a=RANK_KV, rank_b=RANK_AB, BT=BT, USE_A=False, 
    )
    Aak_ref_torch = compute_A_ref_torch(
        operand_a_in=a, operand_b_in=k, g_in=ge_ab, gi_in=gi_ab, rank_a=RANK_AB, rank_b=RANK_KV, BT=BT, USE_A=True, 
    )
    Aab_ref_torch = compute_A_ref_torch(
        operand_a_in=a, operand_b_in=b, g_in=ge_ab, gi_in=gi_ab, rank_a=RANK_AB, rank_b=RANK_AB, BT=BT, USE_A=True, 
    )

    assert torch.allclose(Aqk_ref_torch, Aqk, rtol=5e-2, atol=5e-2)
    assert torch.allclose(Aqb_ref_torch, Aqb, rtol=5e-2, atol=5e-2)
    assert torch.allclose(Aak_ref_torch, Aak, rtol=5e-2, atol=5e-2)
    assert torch.allclose(Aab_ref_torch, Aab, rtol=5e-2, atol=5e-2)

    print("Test passed!")