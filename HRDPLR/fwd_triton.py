from typing import Optional

import torch
import triton
import triton.language as tl

from fla.ops.utils.op import gather
from fla.utils import is_gather_supported, use_cuda_graph

"""
Operator 1 (√)
"""
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8, 16, 32]
        for num_stages in [2, 3, 4, 5]
    ],
    key=['BK', 'BT']
)
@triton.jit(do_not_specialize=['T'])
def chunk_dplr_fwd_A_kernel_intra_sub_intra_rab_generalized(
    q, # [B, T, H, K]
    k, # [B, T, H, K]
    a, # [B, T * RANK_AB, H, K]
    b, # [B, T * RANK_AB, H, K]
    gi, # [B, T, H, K]
    ge, # [B, T, H, K]
    # the following 4 tensors are initialized as empty
    qg, # [B, T, H, K]
    kg, # [B, T, H, K]
    ag, # [B, T * RANK_AB, H, K]
    bg, # [B, T * RANK_AB, H, K] 
    # intermediate results
    Aqk, # [B, T, H, BT]
    Aqb, # [B, T, H, BT * RANK_AB]
    Aab, # [B, T * RANK_AB, H, BT * RANK_AB]
    Aak, # [B, T * RANK_AB, H, BT]
    RANK_AB: tl.constexpr, 
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr, # BT: coarse-grained chunk size along T dim
    BC: tl.constexpr, # BC: fine-grained chunk size along T dim
    BC_AB: tl.constexpr, # BC * RANK_AB
    BK: tl.constexpr,
    NC: tl.constexpr,
):
    """
    Operator 1
    """
    # grid shape: grid = (NT, NC, B * H)
    i_t, i_i, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_j = i_i
    i_b = i_bh // H
    i_h = i_bh % H

    T_start = i_t * BT + i_i * BC 
    if T_start >= T:
        return
    
    o_i_kv = tl.arange(0, BC) # [BC, ]
    o_i_ab = tl.arange(0, BC_AB) # [BC * RANK_AB, ]

    o_k = tl.arange(0, BK) # [BK, ]
    m_k = o_k < K # [BK, ]

    m_A_kv = (i_t * BT + i_i * BC + o_i_kv) < T # [BC, ]
    m_A_ab = (i_t * (BT * RANK_AB) + i_i * (BC * RANK_AB) + o_i_ab) < T * RANK_AB # [BC * RANK_AB, ]
    
    last_idx = min((i_t + 1) * BT, T) - 1 

    # A_qk: [B, T, H, BT]
    # column size: [BC, ]
    o_A_qk = (
        i_b * T * H *  BT 
        + (i_t * BT + i_i * BC + o_i_kv) * H * BT
        + i_h * BT 
        + i_j * BC
    ) # [BC, ]

    # A_qb: [B, T, H, BT * RANK_AB]
    # column size: [BC, ]
    o_A_qb = (
        i_b * T * H * (BT * RANK_AB) 
        + (i_t * BT + i_i * BC + o_i_kv) * H * (BT * RANK_AB)
        + i_h * (BT * RANK_AB) 
        + i_j * (BC * RANK_AB)
    ) # [BC, ]

    # A_ak: [B, T * RANK_AB, H, BT]
    # column size: [BC_AB, ]
    o_A_ak = (
        i_b * (T * RANK_AB) * H * BT 
        + (i_t * (BT * RANK_AB) + i_i * (BC * RANK_AB) + o_i_ab) * H * BT
        + i_h * BT 
        + i_j * BC
    ) # [BC * RANK_AB, ]

    # A_ab: [B, T * RANK_AB, H, BT * RANK_AB]
    # column size: [BC_AB, ]
    o_A_ab = (
        i_b * (T * RANK_AB) * H * (BT * RANK_AB) 
        + (i_t * (BT * RANK_AB) + i_i * (BC * RANK_AB) + o_i_ab) * H * (BT * RANK_AB) 
        + i_h * BT * RANK_AB
        + i_j * (BC * RANK_AB)
    ) # [BC * RANK_AB, ]
    
    # q: [B, T, H, K]
    p_q = tl.make_block_ptr(
        q + i_b * T * H * K + i_h * K, 
        (T, K), 
        (H * K, 1), 
        (i_t * BT + i_i * BC, 0), 
        (BC, BK), 
        (1, 0), 
    ) # [BC, BK]

    # k: [B, T, H, K]
    p_k = tl.make_block_ptr(
        k + i_b * T * H * K + i_h * K, 
        (T, K), 
        (H * K, 1), 
        (i_t * BT + i_i * BC, 0), 
        (BC, BK), 
        (1, 0), 
    ) # [BC, BK]

    # a: [B, T * RANK_AB, H, K]
    p_a = tl.make_block_ptr(
        a + i_b * (T * RANK_AB) * H * K + i_h * K, 
        (T * RANK_AB, K), 
        (H * K, 1), 
        (i_t * (BT * RANK_AB) + i_i * (BC * RANK_AB), 0), 
        (BC_AB, BK), 
        (1, 0), 
    ) # [BC_AB, BK]

    # b: [B, T * RANK_AB, H, K]
    p_b = tl.make_block_ptr(
        b + i_b * (T * RANK_AB) * H * K + i_h * K, 
        (T * RANK_AB, K), 
        (H * K, 1), 
        (i_t * (BT * RANK_AB) + i_i * (BC * RANK_AB), 0), 
        (BC_AB, BK), 
        (1, 0), 
    ) # [BC_AB, BK]

    # gi: [B, T, H, K]
    p_gi = tl.make_block_ptr(
        gi + i_b * T * H * K + i_h * K, 
        (T, K), 
        (H * K, 1), 
        (i_t * BT + i_i * BC, 0), 
        (BC, BK), 
        (1, 0)
    ) # [BC, BK]

    # ge: [B, T, H, K]
    p_ge = tl.make_block_ptr(
        ge + i_b * T * H * K + i_h * K,  
        (T, K), 
        (H * K, 1), 
        (i_t * BT + i_i * BC, 0), 
        (BC, BK), 
        (1, 0)
    ) # [BC, BK]

    p_g_last = gi + i_b * T * H * K + last_idx * H * K + i_h * K + o_k # [BK, ]
    b_g_last = tl.load(p_g_last, mask=m_k, other=0)

    # qg: [B, T, H, K]
    p_qg = tl.make_block_ptr(
        qg + i_b * T * H * K + i_h * K, 
        (T, K), 
        (H * K, 1), 
        (i_t * BT + i_i * BC, 0), 
        (BC, BK), 
        (1, 0)
    ) # [BC, BK]
    
    # kg: [B, T, H, K]
    p_kg = tl.make_block_ptr(
        kg + i_b * T * H * K + i_h * K, 
        (T, K), 
        (H * K, 1), 
        (i_t * BT + i_i * BC, 0), 
        (BC, BK), 
        (1, 0)
    ) # [BC, BK]

    # ag: [B, T * RANK_AB, H, K]
    p_ag = tl.make_block_ptr(
        ag + i_b * (T * RANK_AB) * H * K + i_h * K, 
        (T * RANK_AB, K), 
        (H * K, 1), 
        (i_t * (BT * RANK_AB) + i_i * (BC * RANK_AB), 0), 
        (BC_AB, BK), 
        (1, 0)
    ) # [BC * RANK_AB, BK]

    # bg: [B, T * RANK_AB, H, K]
    p_bg = tl.make_block_ptr(
        bg + i_b * (T * RANK_AB) * H * K + i_h * K, 
        (T * RANK_AB, K), 
        (H * K, 1), 
        (i_t * (BT * RANK_AB) + i_i * (BC * RANK_AB), 0), 
        (BC_AB, BK), 
        (1, 0)
    ) # [BC * RANK_AB, BK]

    b_q = tl.load(p_q, boundary_check=(0, 1)) # [BC, BK]
    b_q = b_q * scale 
    b_k = tl.load(p_k, boundary_check=(0, 1)) # [BC, BK]
    b_a = tl.load(p_a, boundary_check=(0, 1)) # [BC_AB, BK]
    b_b = tl.load(p_b, boundary_check=(0, 1)) # [BC_AB, BK]

    b_gi = tl.load(p_gi, boundary_check=(0, 1)) # [BC, BK]
    b_ge = tl.load(p_ge, boundary_check=(0, 1)) # [BC, BK]

    g_exp = tl.exp(b_gi) # [BC, BK]
    g_exp_inv = tl.exp(-b_gi + b_g_last[None, :]) # [BC, BK] 
    ge_exp = tl.exp(b_ge)

    b_qg = b_q * g_exp # [BC, BK] * [BC, BK] -> [BC, BK]
    b_kg = b_k * g_exp_inv # [BC, BK] * [BC, BK] -> [BC, BK]
    
    if RANK_AB == 2:
        f_ag = tl.interleave(ge_exp.T, ge_exp.T) # [BK, BC * 2]
        f_ag = f_ag.T # [BC * 2, BK]

        f_bg = tl.interleave(g_exp_inv.T, g_exp_inv.T) # [BK, BC * 2]
        f_bg = f_bg.T # [BK, BC * 2]

        b_ag = b_a * f_ag                     # [BC_AB, BK]
        b_bg = b_b * f_bg                     # [BC_AB, BK]
    else:
        b_ag = b_a * ge_exp # [BC, BK] * [BC, BK]
        b_bg = b_b * g_exp_inv # [BC, BK] * [BC, BK]
        
    tl.store(p_qg, b_qg.to(p_qg.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_bg, b_bg.to(p_bg.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_ag, b_ag.to(p_ag.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_kg, b_kg.to(p_kg.dtype.element_ty), boundary_check=(0, 1))

    b_q = b_q.to(b_k.dtype) # [BC, BK]

    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        mask_qk = tl.arange(0, BC) == j # [BC, ]
        b_k_j = tl.sum(tl.where(mask_qk[:, None], b_k, 0), 0) # [BC, BK] -> [BK, ]
        b_gk_j = tl.sum(tl.where(mask_qk[:, None], b_gi, 0), 0) # [BC, BK] -> [BK, ]
        
        tmp_k = tl.exp(b_gi - b_gk_j[None, :]) # [BC, BK]
        
        b_A_qk = tl.sum(
            b_q * b_k_j[None, :] * tmp_k, 1
        ) # [BC, BK] -> [BC, ]
        b_A_qk = tl.where(o_i_kv >= j, b_A_qk, 0.)

        tmp2_k = tl.exp(b_ge - b_gk_j[None, :])
        # if RANK_AB == 2:
        tmp2_b = tl.interleave(tmp2_k.T, tmp2_k.T) # [BK, BC * 2]
        tmp2_b = tmp2_b.T

        # if RANK_AB == 2:
        b_A_ak = tl.sum(b_a * b_k_j[None, :] * tmp2_b, 1) # [BC_AB, ]
        # else:
        #     b_A_ak = tl.sum(b_a * b_k_j[None, :] * tmp2_k, 1) # [BC_AB, ]
        # 
        b_A_ak = tl.where(o_i_ab >= (j + 1) * RANK_AB, b_A_ak, 0.) # [BC_AB, ]
        
        tl.store(Aqk + o_A_qk + j, b_A_qk, mask=m_A_kv)
        tl.store(Aak + o_A_ak + j, b_A_ak, mask=m_A_ab)

        for i_rank in range(RANK_AB):
            mask_ab = tl.arange(0, BC_AB) == j * RANK_AB + i_rank
            b_b_j = tl.sum(tl.where(mask_ab[:, None], b_b, 0), 0) # [BC * RANK_AB, BK] -> [BK, ]

            b_A_qb = tl.sum(b_q * b_b_j[None, :] * tmp_k, 1)
            b_A_qb = tl.where(o_i_kv >= j, b_A_qb, 0.)

            # if RANK_AB == 2:
            b_A_ab = tl.sum(b_a * b_b_j[None, :] * tmp2_b, 1)
            # else:
            #     b_A_ab = tl.sum(b_a * b_b_j[None, :] * tmp2_k, 1)
            b_A_ab = tl.where(o_i_ab >= (j + 1) * RANK_AB, b_A_ab, 0.)

            tl.store(Aqb + o_A_qb + j * RANK_AB + i_rank, b_A_qb, mask=m_A_kv)
            tl.store(Aab + o_A_ab + j * RANK_AB + i_rank, b_A_ab, mask=m_A_ab)

"""
Operator 2 (√)
"""
@triton.autotune(
    configs=[
        triton.Config({'BK': BK}, num_warps=num_warps, num_stages=num_stages)
        for BK in [16, 32, 64, 128]
        for num_warps in [2, 4, 8, 16, 32]
        for num_stages in [2, 3, 4, 5]
    ],
    key=['BC', 'K']
)
@triton.jit(do_not_specialize=['T'])
def chunk_dplr_fwd_A_kernel_intra_sub_inter_rab_generalized(
    q, # [B, T, H, K]
    k, # [B, T, H, K]
    a, # [B, T, H, K]
    b, # [B, T, H, K]
    gi, # [B, T, H, K]
    ge, # [B, T, H, K]
    # intermediate results
    Aqk, # [B, T, H, BT]
    Aqb, # [B, T, H, BT * RANK_AB]
    Aab, # [B, T * RANK_AB, H, BT * RANK_AB]
    Aak, # [B, T * RANK_AB, H, BT]
    RANK_AB: tl.constexpr, 
    scale, 
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BC_AB: tl.constexpr, 
    BK: tl.constexpr,
    NC: tl.constexpr,
):
    # grid shape: grid = (NT, NC * NC, B * H)
    i_t, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b = i_bh // H
    i_h = i_bh % H

    i_i, i_j = i_c // NC, i_c % NC
    
    if i_t * BT + i_i * BC >= T:
        return
    
    if i_i <= i_j: 
        return
    
    b_Aqk = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqb = tl.zeros([BC, BC_AB], dtype=tl.float32)
    b_Aab = tl.zeros([BC_AB, BC_AB], dtype=tl.float32)
    b_Aak = tl.zeros([BC_AB, BC], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK) 
        m_k = o_k < K

        # q: [B, T, H, K]
        p_q = tl.make_block_ptr(
            q + i_b * T * H * K + i_h * K, 
            (T, K), 
            (H * K, 1), 
            (i_t * BT + i_i * BC, i_k * BK), 
            (BC, BK), 
            (1, 0)
        ) # p_q: [BC, BK]

        # a: [B, T * RANK_AB, H, K]
        p_a = tl.make_block_ptr(
            a + i_b * (T * RANK_AB) * H * K + i_h * K, 
            (T * RANK_AB, K), 
            (H * K, 1), 
            (i_t * (BT * RANK_AB) + i_i * (BC * RANK_AB), i_k * BK), 
            (BC_AB, BK), 
            (1, 0), 
        ) # [BC * RANK_AB, BK]
        
        # ge: [B, T, H, K]
        p_ga_e = tl.make_block_ptr(
            ge + i_b * T * H * K + i_h * K, 
            (T, K), 
            (H * K, 1), 
            (i_t * BT + i_i * BC, i_k * BK), 
            (BC, BK), 
            (1, 0)
        ) # [BC, BK]

        # gi: [B, T, H, K]
        p_gq_i = tl.make_block_ptr(
            gi + i_b * T * H * K + i_h * K, 
            (T, K), 
            (H * K, 1), 
            (i_t * BT + i_i * BC, i_k * BK), 
            (BC, BK), 
            (1, 0)
        ) # [BC, BK]
        
        # k: [B, T, H, K]
        p_k = tl.make_block_ptr(
            k + i_b * T * H * K + i_h * K, 
            (K, T), 
            (1, H * K), 
            (i_k * BK, i_t * BT + i_j * BC), 
            (BK, BC), 
            (0, 1)
        ) # [BK, BC]

        # b: [B, T * RANK_AB, H, K]
        p_b = tl.make_block_ptr(
            b + i_b * (T * RANK_AB) * H * K + i_h * K, 
            (K, T * RANK_AB), 
            (1, H * K), 
            (i_k * BK, i_t * (BT * RANK_AB) + i_j * (BC * RANK_AB)), 
            (BK, BC_AB), 
            (0, 1)
        ) # [BK, BC * RANK_AB]

        # gk: [B, T, H, K]
        p_gk = tl.make_block_ptr(
            gi + i_b * T * H * K + i_h * K, 
            (K, T), 
            (1, H * K), 
            (i_k * BK, i_t * BT + i_j * BC), 
            (BK, BC), 
            (0, 1)
        ) # [BK, BC] 

        p_gn = tl.max_contiguous( 
            tl.multiple_of(
                gi + (
                    (i_b * T + i_t * BT + i_i * BC - 1) * H + i_h
                ) * K + o_k,
                BK
            ), 
            BK
        ) # [BK, ]

        # [BK, ]
        b_gn = tl.load(p_gn, mask=m_k, other=0) 
        
        # [BC, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1)) # [BC, BK]
        b_a = tl.load(p_a, boundary_check=(0, 1)) # [BC, BK]

        b_gq_i = tl.load(p_gq_i, boundary_check=(0, 1)) # [BC, BK], inclusive cumsum of query gate
        b_ga_e_rank1 = tl.load(p_ga_e, boundary_check=(0, 1))
        b_ag = tl.zeros([BC_AB, BK], dtype=tl.float32)

        # if RANK_AB == 2:
        b_ga_e = tl.interleave(b_ga_e_rank1.T, b_ga_e_rank1.T) # [BK, BC * 2]
        b_ga_e = b_ga_e.T # [BC * 2, BK]
        
        b_ag = b_a * tl.exp(b_ga_e - b_gn[None, :]) # [BC * 2, BK]
        # else:
        #     b_ag = b_a * tl.exp(b_ga_e_rank1 - b_gn[None, :]) # [BC, BK]
        
        b_qg = b_q * tl.exp(b_gq_i - b_gn[None, :]) * scale # [BC, BK] * ([BC, BK] - [1, BK]) -> [BC, BK]
        
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_b = tl.load(p_b, boundary_check=(0, 1))

        b_gk = tl.load(p_gk, boundary_check=(0, 1))

        tmp_k = tl.exp(b_gn[:, None] - b_gk)
        b_kg = b_k * tmp_k # [BK, BC] * [BK, BC]

        # if RANK_AB == 2:
        tmp_b = tl.interleave(tmp_k, tmp_k) # [BK, BC] -> [BK, BC * 2]
        b_bg = b_b * tmp_b # [BK, BC * 2] * [BK, BC * 2]
        # else:
        #     b_bg = b_b * tmp_k # [BK, BC] * [BK, BC]

        # [BC, BC] using tf32 to improve precision here.
        b_Aab += tl.dot(b_ag, b_bg) # [BC * RANK_AB, BK] @ [BK, BC * RANK_AB] -> [BC * RANK_AB, BC * RANK_AB]
        b_Aak += tl.dot(b_ag, b_kg) # [BC * RANK_AB, BK] @ [BK, BC] -> [BC * RANK_AB, BC]
        b_Aqk += tl.dot(b_qg, b_kg) # [BC, BK] @ [BK, BC] -> [BC, BC]
        b_Aqb += tl.dot(b_qg, b_bg) # [BC, BK] @ [BK, BC * RANK_AB] -> [BC, BC * RANK_AB]

        # Aqk: [B, T, H, BT]
        p_Aqk = tl.make_block_ptr(
            Aqk + i_b * T * H * BT + i_h * BT, 
            (T, BT), 
            (H * BT, 1), 
            (i_t * BT + i_i * BC, i_j * BC), 
            (BC, BC), 
            (1, 0)
        )

        # Aqb: [B, T, H, BT * RANK_AB]
        p_Aqb = tl.make_block_ptr(
            Aqb + i_b * T * H * (BT * RANK_AB) + i_h * (BT * RANK_AB), 
            (T, BT * RANK_AB), 
            (H * BT * RANK_AB, 1), 
            (i_t * BT + i_i * BC, i_j * (BC * RANK_AB)), 
            (BC, BC_AB), 
            (1, 0)
        )

        # Aab: [B, T * RANK_AB, H, BT * RANK_AB]
        p_Aab = tl.make_block_ptr(
            Aab + i_b * (T * RANK_AB) * H * (BT * RANK_AB) + i_h * BT * RANK_AB, 
            (T * RANK_AB, BT * RANK_AB), 
            (H * BT * RANK_AB, 1), 
            (i_t * (BT * RANK_AB) + i_i * (BC * RANK_AB), i_j * (BC * RANK_AB)), 
            (BC_AB, BC_AB), 
            (1, 0)
        )

        # Aak: [B, T * RANK_AB, H, BT]
        p_Aak = tl.make_block_ptr(
            Aak + i_b * (T * RANK_AB) * H * BT + i_h * BT, 
            (T * RANK_AB, BT), 
            (H * BT, 1), 
            (i_t * (BT * RANK_AB) + i_i * (BC * RANK_AB), i_j * BC), 
            (BC_AB, BC), 
            (1, 0)
        )

        tl.store(p_Aqk, b_Aqk.to(Aqk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Aqb, b_Aqb.to(Aqb.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Aab, b_Aab.to(Aab.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Aak, b_Aak.to(Aak.dtype.element_ty), boundary_check=(0, 1))

"""
Operator 3 (Compute the inverse of (I - K @ K.T))
"""
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8, 16]
        for num_stages in [2, 3, 4]
    ],
    key=['BC']
)
@triton.jit(do_not_specialize=['T'])
def fwd_prepare_wy_repr_kernel_chunk32(
    A: torch.Tensor, # [B, T * RANK_AB, H, BT * RANK_AB]
    A_inv: torch.Tensor, # [B, T * RANK_AB, H, BT * RANK_AB]
    B: tl.constexpr, 
    H: tl.constexpr, 
    T: int, 
    BT: tl.constexpr, 
    RANK_AB: tl.constexpr, 
):
    # grid shape: [NT, B * H]
    # each program is responsible for computing the inverse of: 
    # a subchunk of shape [BT * RANK_AB, BT * RANK_AB]
    
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H
    
    # A: [B, T * RANK_AB, H, BT * RANK_AB]
    p_A = tl.make_block_ptr(
        base=A + i_b * (T * RANK_AB) * H * (BT * RANK_AB) + i_h * (BT * RANK_AB), 
        shape=(T * RANK_AB, BT * RANK_AB), 
        strides=(H * (BT * RANK_AB), 1), 
        offsets=(i_t * (BT * RANK_AB), 0), 
        block_shape=(BT * RANK_AB, BT * RANK_AB), 
        order=(1, 0), 
    ) # [BT * RANK_AB, BT * RANK_AB]
    
    # A_inv: [B, T * RANK_AB, H, BT * RANK_AB]
    p_A_inv = tl.make_block_ptr(
        base=A_inv + i_b * (T * RANK_AB) * H * (BT * RANK_AB) + i_h * (BT * RANK_AB), 
        shape=(T * RANK_AB, BT * RANK_AB), 
        strides=(H * (BT * RANK_AB), 1), 
        offsets=(i_t * (BT * RANK_AB), 0), 
        block_shape=(BT * RANK_AB, BT * RANK_AB), 
        order=(1, 0), 
    ) # [BT * RANK_AB, BT * RANK_AB]
    
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_A = tl.where(tl.arange(0, BT * RANK_AB)[:, None] > tl.arange(0, BT * RANK_AB)[None, :], b_A, 0)
    for i in range(RANK_AB, BT * RANK_AB):
        """
        The non-diagonal elements of A in the first RANK_AB rows are all zeros!
        Example (RANK_AB == 2):
        1 0 0 0 0 0 0 0 
        0 1 0 0 0 0 0 0 
        a_{31} a_{32} 1 0 0 0 0 0 
        a_{41} a_{42} 0 1 0 0 0 0 
        ......
        """
        # index the i-th row
        mask = tl.arange(0, BT * RANK_AB) == i
        b_a = tl.sum(tl.where(mask[:, None], b_A, 0), 0)
        
        # compute the i-th row of A_inv
        b_a = b_a + tl.sum(b_a[:, None] * b_A, 0) * (tl.arange(0, BT * RANK_AB) < i)
        b_A = tl.where(mask[:, None], b_a, b_A)

    b_A += tl.arange(0, BT * RANK_AB)[:, None] == tl.arange(0, BT * RANK_AB)[None, :]
    tl.store(p_A_inv, b_A.to(p_A_inv.dtype.element_ty), boundary_check=(0, 1))

@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['BC'],
    use_cuda_graph=use_cuda_graph,
)
@triton.jit(do_not_specialize=['T'])
def prepare_wy_repr_fwd_kernel_chunk64(
    A_ab,
    A_ab_inv,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    GATHER_SUPPORTED: tl.constexpr = is_gather_supported
):
    """
    Directly borrowed from fla.
    """
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    p_A1 = tl.make_block_ptr(A_ab + (bos*H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT, 0), (BC, BC), (1, 0))
    p_A2 = tl.make_block_ptr(A_ab + (bos*H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT + BC, BC), (BC, BC), (1, 0))
    p_A3 = tl.make_block_ptr(A_ab + (bos*H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT + BC, 0), (BC, BC), (1, 0))
    p_A_inv1 = tl.make_block_ptr(A_ab_inv + (bos*H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT, 0), (BC, BC), (1, 0))
    p_A_inv2 = tl.make_block_ptr(A_ab_inv + (bos*H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT + BC, BC), (BC, BC), (1, 0))
    p_A_inv3 = tl.make_block_ptr(A_ab_inv + (bos*H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT + BC, 0), (BC, BC), (1, 0))
    p_A_inv4 = tl.make_block_ptr(A_ab_inv + (bos*H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT, BC), (BC, BC), (1, 0))

    b_A = tl.load(p_A1, boundary_check=(0, 1))
    b_A2 = tl.load(p_A2, boundary_check=(0, 1))
    b_A3 = tl.load(p_A3, boundary_check=(0, 1))
    b_A = tl.where(tl.arange(0, BC)[:, None] > tl.arange(0, BC)[None, :], b_A, 0)
    b_A2 = tl.where(tl.arange(0, BC)[:, None] > tl.arange(0, BC)[None, :], b_A2, 0)

    for i in range(1, BC):
        if GATHER_SUPPORTED:
            row_idx = tl.full([1, BC], i, dtype=tl.int16)
            # [1, BK] -> [BK]
            b_a = tl.sum(gather(b_A, row_idx, axis=0), 0)
            b_a2 = tl.sum(gather(b_A2, row_idx, axis=0), 0)
        else:
            mask = tl.arange(0, BC) == i
            b_a = tl.sum(tl.where(mask[:, None], b_A, 0), 0)
            b_a2 = tl.sum(tl.where(mask[:, None], b_A2, 0), 0)
        mask = tl.arange(0, BC) == i
        # b_a = tl.sum(tl.where(mask[:, None], b_A, 0), 0)
        # b_a2 = tl.sum(tl.where(mask[:, None], b_A2, 0), 0)
        b_a = b_a + tl.sum(b_a[:, None] * b_A, 0) * (tl.arange(0, BC) < i)
        b_a2 = b_a2 + tl.sum(b_a2[:, None] * b_A2, 0) * (tl.arange(0, BC) < i)
        b_A = tl.where(mask[:, None], b_a, b_A)
        b_A2 = tl.where(mask[:, None], b_a2, b_A2)

    # blockwise computation of lower triangular matrix's inverse
    # i.e., [A11, 0; A21, A22]^-1 = [A11^-1, 0; -A22^-1 A21 A11^-1, A22^-1]
    b_A += tl.arange(0, BC)[:, None] == tl.arange(0, BC)[None, :]
    b_A2 += tl.arange(0, BC)[:, None] == tl.arange(0, BC)[None, :]
    b_A3 = tl.dot(tl.dot(b_A2, b_A3), b_A)
    # tl.debug_barrier()
    tl.store(p_A_inv1, b_A.to(p_A_inv1.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_A_inv2, b_A2.to(p_A_inv2.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_A_inv3, b_A3.to(p_A_inv3.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    # causal mask
    tl.store(p_A_inv4, tl.zeros([BC, BC], dtype=tl.float32).to(p_A_inv4.dtype.element_ty), boundary_check=(0, 1))



"""
Operator 4 (Compute w, u according to formula (19), (20) in HDLA paper)
"""
@triton.autotune(
    configs=[
        triton.Config({'BK': BK, 'BV': BV}, num_warps=num_warps)
        for num_warps in [2, 4, 8, 16]
        for num_stages in [2, 3, 4]
        for BK in [16, 32, 64]
        for BV in [16, 32, 64]
    ],
    key=['BT']
)
@triton.jit(do_not_specialize=['T'])
def fwd_wu_kernel(
    u, # [B, T * RANK_AB, H, V]
    w, # [B, T * RANK_AB, H, K]
    ag, # [B, T * RANK_AB, H, K]
    v, # [B, T, H, V]
    A_ab_inv, # [B, T * RANK_AB, H, BT * RANK_AB]
    A_ak, # [B, T * RANK_AB, H, BT]
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    RANK_AB: tl.constexpr, 
    BT_AB: tl.constexpr, 
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    # grid shape: [NT, B * H]
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H

    # A_ab_inv: [B, T * RANK_AB, H, BT * RANK_AB]
    p_A_ab_inv = tl.make_block_ptr(
        A_ab_inv + i_b * (T * RANK_AB) * H * (BT * RANK_AB) + i_h * BT * RANK_AB, 
        (T * RANK_AB, BT * RANK_AB), 
        (H * (BT * RANK_AB), 1), 
        (i_t * (BT * RANK_AB), 0), 
        (BT_AB, BT_AB), 
        (1, 0)
    )

    # A_ak: [B, T * RANK_AB, H, BT]
    p_A_ak = tl.make_block_ptr(
        base=A_ak + i_b * (T * RANK_AB) * H * BT + i_h * BT, 
        shape=(T * RANK_AB, BT), 
        strides=(H * BT, 1), 
        offsets=(i_t * (BT * RANK_AB), 0), 
        block_shape=(BT_AB, BT), 
        order=(1, 0), 
    ) # [BT * RANK_AB, BT]
    
    b_Aab_inv = tl.load(p_A_ab_inv, boundary_check=(0, 1))
    b_Aak = tl.load(p_A_ak, boundary_check=(0, 1))
    
    o_s_kv = tl.arange(0, BT)
    o_s_ab = tl.interleave(o_s_kv, o_s_kv)

    o_i_ab = tl.arange(0, BT_AB)

    b_Aab_inv = tl.where(o_i_ab[:, None] >= o_i_ab[None, :], b_Aab_inv, 0) # tril(Aab, 0)
    
    b_Aak = tl.where(o_s_ab[:, None] > o_s_kv[None, :], b_Aak, 0) # tril(Aak, -1)

    b_Aak = tl.dot(b_Aab_inv, b_Aak, allow_tf32=True)
    
    b_Aak = b_Aak.to(u.dtype.element_ty)
    b_Aab_inv = b_Aab_inv.to(u.dtype.element_ty)

    for i_k in range(tl.cdiv(K, BK)):
        # ag: [B, T * RANK_AB, H, K]
        p_ag = tl.make_block_ptr(
            ag + i_b * (T * RANK_AB) * H * K + i_h * K, 
            (T * RANK_AB, K), 
            (H * K, 1), 
            (i_t * (BT * RANK_AB), i_k * BK), 
            (BT_AB, BK), 
            (1, 0)
        )
        # w: [B, T * RANK_AB, H, K]
        p_w = tl.make_block_ptr(
            w + i_b * (T * RANK_AB) * H * K + i_h * K, 
            (T * RANK_AB, K), 
            (H * K, 1), 
            (i_t * (BT * RANK_AB), i_k * BK), 
            (BT_AB, BK), 
            (1, 0)
        )

        b_ag = tl.load(p_ag, boundary_check=(0, 1))
        b_w = tl.dot(b_Aab_inv.to(b_ag.dtype), b_ag, allow_tf32=False)
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))

    for i_v in range(tl.cdiv(V, BV)):
        # v: [B, T, H, V]
        p_v = tl.make_block_ptr(
            v + i_b * T * H * V + i_h * V, 
            (T, V), 
            (H * V, 1), 
            (i_t * BT, i_v * BV), 
            (BT, BV), 
            (1, 0)
        )

        # u: [B, T * RANK_AB, H, V]
        p_u = tl.make_block_ptr(
            u + i_b * (T * RANK_AB) * H * V + i_h * V, 
            (T * RANK_AB, V), 
            (H * V, 1), 
            (i_t * (BT * RANK_AB), i_v * BV), 
            (BT_AB, BV), 
            (1, 0)
        )
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_u = tl.dot(b_Aak.to(b_v.dtype), b_v, allow_tf32=False)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))

"""
Operator 5 (Compute formula (24) in HDLA paper)
"""
@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8, 16, 32]
        for num_stages in [2, 3, 4, 5]
    ],
    key=['BT', 'BK', 'BV']
)
@triton.jit(do_not_specialize=['T'])
def chunk_dplr_fwd_kernel_h(
    kg, # [B, T, H, K]
    v, # [B, T, H, V]        
    w, # [B, T * RANK_AB, H, K]
    bg, # [B, T * RANK_AB, H, K]
    u, # [B, T * RANK_AB, H, V]
    v_new, # [B, T * RANK_AB, H, V]
    gk, # [B, T, H, K]
    h, # [B, NT, H, K, V]
    h0, # [B, H, K, V]
    ht, # [B, H, K, V]
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    RANK_AB: tl.constexpr, 
    BC_AB: tl.constexpr, 
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
):
    # grid shape: [NK, NV, B * H]
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b = i_nh // H  
    i_h = i_nh % H  
    
    # [BK, BV]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)

    if USE_INITIAL_STATE:
        # h0: [B, H, K, V]
        p_h0 = tl.make_block_ptr(
            h0 + i_b * H * K * V + i_h * K * V, 
            (K, V), 
            (V, 1), 
            (i_k * BK, i_v * BV), 
            (BK, BV), 
            (1, 0)
        )
        b_h = tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)

    for i_t in range(NT):
        # h: [B, NT, H, K, V]
        p_h = tl.make_block_ptr(
            h + i_b * NT * H * K * V + i_t * H * K * V + i_h * K * V, 
            (K, V), 
            (V, 1), 
            (i_k * BK, i_v * BV), 
            (BK, BV), 
            (1, 0)
        )
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))

        b_hc = tl.zeros([BK, BV], dtype=tl.float32)
        
        for i_c in range(tl.cdiv(BT, BC)):
            
            # kg: [B, T, H, K]
            p_kg = tl.make_block_ptr(
                kg + i_b * T * H * K + i_h * K, 
                (T, K),  
                (H * K, 1),        
                (i_t * BT + i_c * BC, i_k * BK), 
                (BC, BK), 
                (1, 0)
            )
            
            # bg: [B, T * RANK_AB, H, K]
            p_bg = tl.make_block_ptr(
                bg + i_b * T * RANK_AB * H * K + i_h * K,  
                (T * RANK_AB, K), 
                (H * K, 1),                  
                (i_t * BT * RANK_AB + i_c * BC * RANK_AB, i_k * BK),  
                (BC * RANK_AB, BK), 
                (1, 0)
            )
            
            # w: [B, T * RANK_AB, H, K]
            p_w = tl.make_block_ptr(
                w + i_b * T * RANK_AB * H * K + i_h * K,  
                (T * RANK_AB, K),  
                (H * K, 1),                 
                (i_t * BT * RANK_AB + i_c * BC * RANK_AB, i_k * BK), 
                (BC * RANK_AB, BK), 
                (1, 0)
            )
            
            # v: [B, T, H, V]
            p_v = tl.make_block_ptr(
                v + i_b * T * H * V + i_h * V,
                (T, V), 
                (H * V, 1),        
                (i_t * BT + i_c * BC, i_v * BV), 
                (BC, BV), 
                (1, 0)
            )
            
            # u: [B, T * RANK_AB, H, V]
            p_u = tl.make_block_ptr(
                u + i_b * T * RANK_AB * H * V + i_h * V,  
                (T * RANK_AB, V),  
                (H * V, 1),                  
                (i_t * BT * RANK_AB + i_c * BC * RANK_AB, i_v * BV),  
                (BC * RANK_AB, BV), 
                (1, 0)
            )
            
            # v_new: [B, T * RANK_AB, H, V]
            p_v_new = tl.make_block_ptr(
                v_new + i_b * T * RANK_AB * H * V + i_h * V,  
                (T * RANK_AB, V),  
                (H * V, 1),  
                (i_t * BT * RANK_AB + i_c * BC * RANK_AB, i_v * BV),
                (BC * RANK_AB, BV), 
                (1, 0)
            )
            
            b_kg = tl.load(p_kg, boundary_check=(0, 1)) 
            b_v = tl.load(p_v, boundary_check=(0, 1)) 
            b_w = tl.load(p_w, boundary_check=(0, 1)) 
            b_bg = tl.load(p_bg, boundary_check=(0, 1))  
            b_u = tl.load(p_u, boundary_check=(0, 1)) 
            
            b_v2 = tl.dot(b_w, b_h.to(b_w.dtype)) + b_u
            
            b_hc += tl.dot(b_kg.T, b_v)
            b_hc += tl.dot(b_bg.T, b_v2.to(b_bg.dtype))

            tl.store(p_v_new, b_v2.to(p_v_new.dtype.element_ty), boundary_check=(0, 1))


        last_t_in_block = min((i_t + 1) * BT - 1, T - 1)
        # gk: [B, T, H, K]
        p_gk_last = tl.make_block_ptr(
            gk + i_b * T * H * K + last_t_in_block * H * K + i_h * K, 
            (BK,), 
            (1,), 
            (i_k * BK,), 
            (BK,), 
            (0,)
        )
        b_g_last = tl.load(p_gk_last, boundary_check=(0,)).to(tl.float32)
        
        b_h *= tl.exp(b_g_last[:, None]) 
        b_h += b_hc

    if STORE_FINAL_STATE:
        # ht: [B, H, K, V]
        p_ht = tl.make_block_ptr(
            ht + i_b * H * K * V + i_h * K * V,  
            (K, V), 
            (V, 1), 
            (i_k * BK, i_v * BV), 
            (BK, BV), 
            (1, 0)
        )
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))

"""
Operator 6
"""
@triton.autotune(
    configs=[
        triton.Config({'BK': BK, 'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64, 128]
        for BV in [32, 64, 128]
        for num_warps in [2, 4, 8, 16, 32]
        for num_stages in [2, 3, 4]
    ],
    key=['BT'],
)
@triton.jit(do_not_specialize=['T'])
def chunk_dplr_fwd_kernel_o(
    qg, # [B, T, H, K]
    v, # [B, T, H, V]
    v_new, # [B, T * RANK_AB, H, V]
    A_qk, # [B, T, H, BT]
    A_qb, # [B, T, H, BT * RANK_AB]
    h, # [B, NT, H, K, V]
    o, # [B, T, H, V]
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    RANK_AB: tl.constexpr, 
    BT_AB: tl.constexpr, # BT_AB = BT * RANK_AB
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    NT = tl.cdiv(T, BT)

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        
        # qg: [B, T, H, K]
        p_qg = tl.make_block_ptr(
            qg + i_b * T * H * K + i_h * K, 
            (T, K), 
            (H * K, 1), 
            (i_t * BT, i_k * BK), 
            (BT, BK), 
            (1, 0)
        )

        # h: [B, NT, H, K, V]
        p_h = tl.make_block_ptr(
            h + i_b * NT * H * K * V + i_t * H * K * V + i_h * K * V, 
            (K, V), 
            (V, 1), 
            (i_k * BK, i_v * BV), 
            (BK, BV), 
            (1, 0)
        )
        
        b_qg = tl.load(p_qg, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_o += tl.dot(b_qg, b_h)

    # A_qk: [B, T, H, BT]
    p_Aqk = tl.make_block_ptr(
        A_qk + i_b * T * H * BT + i_h * BT, 
        (T, BT), 
        (H * BT, 1), 
        (i_t * BT, 0), 
        (BT, BT), 
        (1, 0)
    ) # [BT, BT]
    
    # A_qb: [B, T, H, BT * RANK_AB]
    p_Aqb = tl.make_block_ptr(
        A_qb + i_b * T * H * BT_AB + i_h * BT_AB, 
        (T, BT * RANK_AB), 
        (H * BT_AB, 1), 
        (i_t * BT, 0), 
        (BT, BT_AB), 
        (1, 0)
    ) # [BT, BT_AB]
    
    # v: [B, T, H, V]
    p_v = tl.make_block_ptr(
        v + i_b * T * H * V + i_h * V, 
        (T, V), 
        (H * V, 1), 
        (i_t * BT, i_v * BV),
        (BT, BV), 
        (1, 0)
    )
    
    # v_new: [B, T * RANK_AB, H, V]
    p_v_new = tl.make_block_ptr(
        v_new + i_b * T * RANK_AB * H * V + i_h * V, 
        (T * RANK_AB, V), 
        (H * V, 1), 
        (i_t * BT * RANK_AB, i_v * BV), 
        (BT * RANK_AB, BV), 
        (1, 0)
    )
    
    # o: [B, T, H, V]
    p_o = tl.make_block_ptr(
        o + i_b * T * H * V + i_h * V, 
        (T, V), 
        (H * V, 1), 
        (i_t * BT, i_v * BV), 
        (BT, BV), 
        (1, 0)
    )

    o_s_kv = tl.arange(0, BT) # [BT]
    o_s_ab = tl.interleave(o_s_kv, o_s_kv) # [BT_AB]

    m_s_qk = o_s_kv[:, None] >= o_s_kv[None, :] # [BT, BT]
    m_s_qb = o_s_kv[:, None] >= o_s_ab[None, :] # [BT, BT_AB]
    b_Aqk = tl.load(p_Aqk, boundary_check=(0, 1))
    b_Aqb = tl.load(p_Aqb, boundary_check=(0, 1))
    b_Aqk = tl.where(m_s_qk, b_Aqk, 0)
    b_Aqb = tl.where(m_s_qb, b_Aqb, 0)
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_v_new = tl.load(p_v_new, boundary_check=(0, 1))
    b_o = b_o + tl.dot(b_Aqk.to(b_v.dtype), b_v) + tl.dot(b_Aqb.to(b_v_new.dtype), b_v_new)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))