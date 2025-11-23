from typing import Optional

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp, gather
from fla.utils import check_shared_mem, is_amd, is_gather_supported, use_cuda_graph

NUM_WARPS_AUTOTUNE = [2, 4, 8, 16] if is_amd else [2, 4, 8, 16, 32]
BK_LIST = [32, 64, 128] if check_shared_mem() else [16, 32]

"""
Test passed
"""
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS_AUTOTUNE
        for num_stages in [2, 3, 4]
    ],
    key=['BV', 'BT'],
    use_cuda_graph=use_cuda_graph,
)
@triton.jit(do_not_specialize=['T'])
def chunk_dplr_bwd_kernel_dAu(
    v, # [B, T, H, V]
    do, # [B, T, H, V]
    v_new, # [B, T * RANK_AB, H, V]
    A_qb, # [B, T, H, BT * RANK_AB]
    dA_qk, # [B, T, H, BT]
    dA_qb, # [B, T, H, BT * RANK_AB]
    dv_new, # [B, T * RANK_AB, H, V]
    scale: tl.constexpr,
    T,
    RANK_AB: tl.constexpr, 
    BT_AB: tl.constexpr, 
    H: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
):
    # grid shape: [NT, B * H]
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    bos, eos = i_b * T, i_b * T + T
    T = eos - bos

    b_dA_qk = tl.zeros([BT, BT], dtype=tl.float32)

    b_dA_qb = tl.zeros([BT, BT * RANK_AB], dtype=tl.float32)
    # [B, T, H, BT * RANK_AB]
    p_A_qb = tl.make_block_ptr(
        base=A_qb + (bos * H + i_h) * BT_AB, shape=(T, BT_AB), strides=(H * BT_AB, 1), 
        offsets=(i_t * BT, 0), block_shape=(BT, BT_AB), order=(1, 0), 
    ) # [BT, BT_AB]

    b_A_qb = tl.load(p_A_qb, boundary_check=(0, 1)) # [BT, BT_AB]

    o_qk = tl.arange(0, BT)
    o_ab = tl.interleave(o_qk, o_qk)

    for i_v in range(tl.cdiv(V, BV)):
        p_do = tl.make_block_ptr(do + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_v = tl.make_block_ptr(v + (bos*H + i_h) * V, (V, T), (1, H*V), (i_v * BV, i_t * BT), (BV, BT), (0, 1))
        # [B, T * RANK_AB, H, V]
        p_v_new = tl.make_block_ptr(
            v_new + (bos * RANK_AB * H + i_h) * V, (V, T * RANK_AB), (1, H * V), 
            (i_v * BV, i_t * BT * RANK_AB), (BV, BT * RANK_AB), (0, 1)
        ) # [BV, BT_AB]
        # [B, T * RANK_AB, H, V]
        p_dv_new = tl.make_block_ptr(
            dv_new + (bos * RANK_AB * H + i_h) * V, (T * RANK_AB, V), (H * V, 1), 
            (i_t * BT * RANK_AB, i_v * BV), (BT * RANK_AB, BV), (1, 0)
        ) # [BT_AB, BV]

        b_v = tl.load(p_v, boundary_check=(0, 1)) # [BV, BT]
        b_do = tl.load(p_do, boundary_check=(0, 1)) # [BT, BV]
        b_v_new = tl.load(p_v_new, boundary_check=(0, 1)) # [BV, BT_AB]

        b_dA_qk += tl.dot(b_do, b_v) # [BT, BV] @ [BV, BT] -> [BT, BT]
        b_dA_qb += tl.dot(b_do, b_v_new) # [BT, BV] @ [BV, BT_AB] -> [BT, BT_AB]

        b_dv_new = tl.dot(tl.trans(b_A_qb), b_do) # [BT_AB, BT] @ [BT, BV] -> [BT_AB, BV]
        tl.store(p_dv_new, b_dv_new.to(p_dv_new.dtype.element_ty), boundary_check=(0, 1))

    p_dA_qk = tl.make_block_ptr(dA_qk + (bos * H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    # [B, T, H, BT * RANK_AB]
    p_dA_qb = tl.make_block_ptr(
        dA_qb + (bos * H + i_h) * BT_AB, (T, BT_AB), (H * BT_AB, 1), (i_t * BT, 0), 
        (BT, BT_AB), (1, 0)
    )

    m_qk = o_qk[:, None] >= o_qk[None, :]
    m_qb = o_qk[:, None] >= o_ab[None, :]

    b_dA_qk = tl.where(m_qk, b_dA_qk * scale, 0.)
    tl.store(p_dA_qk, b_dA_qk.to(p_dA_qk.dtype.element_ty), boundary_check=(0, 1))
    b_dA_qb = tl.where(m_qb, b_dA_qb * scale, 0.)
    tl.store(p_dA_qb, b_dA_qb.to(p_dA_qb.dtype.element_ty), boundary_check=(0, 1))

"""
Test passed
"""
@triton.heuristics({
    'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None,
    'USE_INITIAL_STATE': lambda args: args['dh0'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS_AUTOTUNE
        for num_stages in [2, 3, 4]
    ],
    key=['BT', 'BK', 'BV', "V"],
    use_cuda_graph=use_cuda_graph,
)
@triton.jit(do_not_specialize=['T'])
def chunk_dplr_bwd_kernel_dhu(
    qg, # [B, T, H, K]
    bg, # [B, T * RANK_AB, H, K]
    w, # [B, T * RANK_AB, H, K]
    gk, # [B, T, H, K]
    dht, # [B, H, K, V]
    dh0, # [B, H, K, V]
    do, # [B, T, H, V]
    dh, # [B, NT, H, K, V]
    dv, # (dv_new_intra) [B, T * RANK_AB, H, V]
    dv2, # (dv_new) [B, T * RANK_AB, H, V]
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    RANK_AB: tl.constexpr, 
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
):
    # grid shape: [NK, NV, B * H]
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_h = i_nh // H, i_nh % H

    bos, eos = i_n * T, i_n * T + T
    NT = tl.cdiv(T, BT)
    boh = i_n * NT

    b_dh = tl.zeros([BK, BV], dtype=tl.float32) # [BK, BV]
    if USE_FINAL_STATE_GRADIENT:
        # dht: [B, H, K, V]
        p_dht = tl.make_block_ptr(dht + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_dh += tl.load(p_dht, boundary_check=(0, 1)).to(b_dh.dtype)

    mask_k = tl.arange(0, BK) < K
    for i_t in range(NT - 1, -1, -1):
        # store the gradient of  H_{[t + 1]}, i.e. the hidden state after the i_t-th block ends
        # dh: [B, NT, H, K, V]
        p_dh = tl.make_block_ptr(dh + ((boh + i_t) * H + i_h) * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_dh, b_dh.to(p_dh.dtype.element_ty), boundary_check=(0, 1))
        b_dh_tmp = tl.zeros([BK, BV], dtype=tl.float32)
        for i_c in range(tl.cdiv(BT, BC) - 1, -1, -1):
            # qg: [B, T, H, K]
            p_qg = tl.make_block_ptr(qg + (bos*H+i_h)*K, (K, T), (1, H*K), (i_k * BK, i_t * BT + i_c * BC), (BK, BC), (0, 1))
            # do: [B, T, H, K]
            p_do = tl.make_block_ptr(do+(bos*H+i_h)*V, (T, V), (H*V, 1), (i_t*BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))
            
            b_qg = tl.load(p_qg, boundary_check=(0, 1)) # [BK, BC]
            b_do = tl.load(p_do, boundary_check=(0, 1)) # [BC, BV]            

            for i_r in range(RANK_AB):
                # dv: [B, T * RANK_AB, H, V]
                p_dv = tl.make_block_ptr(
                    dv + (bos * RANK_AB * H + i_h) * V, (T * RANK_AB, V), 
                    (H * V, 1), (i_t * BT * RANK_AB + i_c * BC * RANK_AB + i_r * BC, i_v * BV), 
                    (BC, BV), (1, 0)
                ) # [BC, BV]
                # bg: [B, T * RANK_AB, H, K]
                p_bg = tl.make_block_ptr(
                    bg + (bos * RANK_AB * H + i_h)*K, (T * RANK_AB, K), (H * K, 1), 
                    (i_t * BT * RANK_AB + i_c * BC * RANK_AB + i_r * BC, i_k * BK), 
                    (BC, BK), (1, 0)
                ) # [BC, BK]
                # w: [B, T * RANK_AB, H, K]
                p_w = tl.make_block_ptr(
                    w + (bos * RANK_AB * H + i_h) * K, (K, T * RANK_AB), (1, H * K), 
                    (i_k * BK, i_t * BT * RANK_AB + i_c * BC * RANK_AB + i_r * BC), 
                    (BK, BC), (0, 1)
                ) # [BK, BC]
                # dv2: [B, T * RANK_AB, H, V]
                p_dv2 = tl.make_block_ptr(
                    dv2 + (bos * RANK_AB * H + i_h) * V, (T * RANK_AB, V), (H * V, 1), 
                    (i_t * BT * RANK_AB + i_c * BC * RANK_AB + i_r * BC, i_v * BV),
                    (BC, BV), (1, 0)
                ) # [BC, BV]
                b_bg = tl.load(p_bg, boundary_check=(0, 1)) # [BC, BK]
                b_w = tl.load(p_w, boundary_check=(0, 1)) # [BK, BC]
                b_dv = tl.load(p_dv, boundary_check=(0, 1)) # [BC, BV]
                
                b_dv2 = b_dv + tl.dot(b_bg, b_dh.to(b_bg.dtype)) # [BC, BV]
                tl.store(p_dv2, b_dv2.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
                b_dh_tmp += tl.dot(b_w, b_dv2.to(b_qg.dtype)) # [BK, BC] @ [BC, BV] -> [BC, BV]

            b_dh_tmp += tl.dot(b_qg, b_do.to(b_qg.dtype)) # [BK, BC] @ [BC, BV] -> [BK, BV]
            
        last_idx = min((i_t + 1) * BT, T) - 1
        bg_last = tl.load(gk + ((bos + last_idx) * H + i_h) * K + tl.arange(0, BK), mask=mask_k)
        b_dh *= exp(bg_last)[:, None]
        b_dh += b_dh_tmp

    if USE_INITIAL_STATE:
        p_dh0 = tl.make_block_ptr(dh0 + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), boundary_check=(0, 1))

"""
Test passed
"""
@triton.autotune(
    configs=[
        triton.Config({'BK': BK, 'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS_AUTOTUNE
        for num_stages in [2, 3, 4]
        for BK in BK_LIST
        for BV in BK_LIST
    ],
    key=['BT'],
    use_cuda_graph=use_cuda_graph,
)
@triton.jit
def chunk_dplr_bwd_kernel_dv(
    A_qk, # [B, T, H, BT]
    kg, # [B, T, H, K]
    do, # [B, T, H, V]
    dv, # [B, T, H, V]
    dh, # [B, NT, H, K, V]
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    """
    This kernel is directly borrowed from:
    https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/generalized_delta_rule/dplr/chunk_o_bwd.py
    """
    # grid shape: [NV, NT, B * H]
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    NT = tl.cdiv(T, BT)
    i_tg = i_b * NT + i_t
    bos, eos = i_b * T, i_b * T + T

    b_dv = tl.zeros([BT, BV], dtype=tl.float32) # [BT, BV]

    A_qk += (bos * H + i_h) * BT
    do += (bos * H + i_h) * V
    dv += (bos * H + i_h) * V
    kg += (bos * H + i_h) * K
    dh += (i_tg * H + i_h) * K*V

    stride_qk = H * K
    stride_vo = H * V
    stride_A = H * BT

    for i_k in range(tl.cdiv(K, BK)):
        p_dh = tl.make_block_ptr(dh, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        p_kg = tl.make_block_ptr(kg, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_dh = tl.load(p_dh, boundary_check=(0, 1))
        b_kg = tl.load(p_kg, boundary_check=(0, 1))
        b_dv += tl.dot(b_kg, b_dh.to(b_kg.dtype))

    p_Aqk = tl.make_block_ptr(A_qk, (BT, T), (1, stride_A), (0, i_t * BT), (BT, BT), (0, 1))
    b_A = tl.where(tl.arange(0, BT)[:, None] <= tl.arange(0, BT)[None, :], tl.load(p_Aqk, boundary_check=(0, 1)), 0)
    p_do = tl.make_block_ptr(do, (T, V), (stride_vo, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_dv = tl.make_block_ptr(dv, (T, V), (stride_vo, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_dv += tl.dot(b_A.to(b_do.dtype), b_do)
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))

"""
Test passed
"""
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS_AUTOTUNE
        for num_stages in [2, 3, 4]
    ],
    key=['BT', 'BK', 'BV'],
    use_cuda_graph=use_cuda_graph,
)
@triton.jit
def chunk_dplr_bwd_o_kernel(
    v, # [B, T, H, V]
    v_new, # [B, T * RANK_AB, H, V]
    h, # [B, NT, H, K, V]
    do, # [B, T, H, K]
    dh, # [B, NT, H, K, V]
    dk,
    db,
    w,
    dq,
    dv, # (dv_new) [B, T * RANK_AB, H, K]
    dw,
    gk,
    dgk_last,
    k,
    b,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    RANK_AB: tl.constexpr, 
    BT: tl.constexpr,
    BT_AB: tl.constexpr, 
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    NT = tl.cdiv(T, BT)
    i_tg = i_b * NT + i_t
    bos, eos = i_b * T, i_b * T + T

    # offset calculation
    # v: [B, T, H, V]
    v += (bos * H + i_h) * V
    # v_new: [B, T * RANK_AB, H, V]
    v_new += (bos * RANK_AB * H + i_h) * V
    # do: [B, T, H, V]
    do += (bos * H + i_h) * V
    # h: [B, NT, H, K, V]
    h += (i_tg * H + i_h) * K * V
    # dh: [B, NT, H, K, V]
    dh += (i_tg * H + i_h) * K * V
    # dk: [B, T, H, K]
    dk += (bos * H + i_h) * K
    # k: [B, T, H, K]
    k += (bos * H + i_h) * K
    # db: [B, T * RANK_AB, H, K]
    db += (bos * RANK_AB * H + i_h) * K
    # b: [B, T * RANK_AB, H, K]
    b += (bos * RANK_AB * H + i_h) * K
    # dw: [B, T * RANK_AB, H, K]
    dw += (bos * RANK_AB * H + i_h) * K
    # dv: [B, T * RANK_AB, H, V]
    dv += (bos * RANK_AB * H + i_h) * V
    # dq: [B, T, H, K]
    dq += (bos * H + i_h) * K
    # w: [B, T * RANK_AB, H, K]
    w += (bos * RANK_AB * H + i_h) * K

    dgk_last += (i_tg * H + i_h) * K
    gk += (bos * H + i_h) * K

    stride_qk = H * K
    stride_vo = H * V

    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_dw_1 = tl.zeros([BT, BK], dtype=tl.float32)
    b_dw_2 = tl.zeros([BT, BK], dtype=tl.float32)
    b_db_1 = tl.zeros([BT, BK], dtype=tl.float32)
    b_db_2 = tl.zeros([BT, BK], dtype=tl.float32)
    b_dgk_last = tl.zeros([BK], dtype=tl.float32)

    for i_v in range(tl.cdiv(V, BV)):
        # v: [B, T, H, V]
        p_v = tl.make_block_ptr(v, (T, V), (stride_vo, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        # do: [B, T, H, V]
        p_do = tl.make_block_ptr(do, (T, V), (stride_vo, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        # h: [B, NT, H, K, V]
        p_h = tl.make_block_ptr(h, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        # dh: [B, NT, H, K, V]
        p_dh = tl.make_block_ptr(dh, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        
        b_v = tl.load(p_v, boundary_check=(0, 1)) # [BT, BV]
        b_do = tl.load(p_do, boundary_check=(0, 1)) # [BT, BV]
        b_h = tl.load(p_h, boundary_check=(0, 1)) # [BV, BK]
        b_dh = tl.load(p_dh, boundary_check=(0, 1)) # [BV, BK]
        b_dgk_last += tl.sum((b_h * b_dh).to(tl.float32), axis=0) # [BV, BK] -> [BK, ]

        # v_new: [B, T * RANK_AB, H, V]
        p_v_new_1 = tl.make_block_ptr(
            v_new, (T * RANK_AB, V), (stride_vo, 1), (i_t * BT * RANK_AB, i_v * BV), 
            (BT, BV), (1, 0)
        ) # [BT, BV]
        p_v_new_2 = tl.make_block_ptr(
            v_new, (T * RANK_AB, V), (stride_vo, 1), (i_t * BT * RANK_AB + BT, i_v * BV), 
            (BT, BV), (1, 0)
        ) # [BT, BV]
        b_v_new_1 = tl.load(p_v_new_1, boundary_check=(0, 1)) # [BT, BV]
        b_v_new_2 = tl.load(p_v_new_2, boundary_check=(0, 1)) # [BT, BV]

        b_dq += tl.dot(b_do, b_h.to(b_do.dtype)) # [BT, BV] @ [BV, BK] -> [BT, BK]
        b_dk += tl.dot(b_v, b_dh.to(b_v.dtype)) # [BT, BV] @ [BV, BK] -> [BT, BK]
        """
        |A1| @ H = |A1 @ H|
        |A2|       |A2 @ H|
        """
        b_db_1 += tl.dot(b_v_new_1, b_dh.to(b_v_new_1.dtype)) # [BT, BV] @ [BV, BK] -> [BT, BK]
        b_db_2 += tl.dot(b_v_new_2, b_dh.to(b_v_new_1.dtype)) # [BT, BV] @ [BV, BK] -> [BT, BK]
        
        # dv: [B, T * RANK_AB, H, K]
        p_dv_1 = tl.make_block_ptr(
            dv, (T * RANK_AB, V), (stride_vo, 1), (i_t * BT * RANK_AB, i_v * BV), 
            (BT, BV), (1, 0)
        ) # [BT, BV]
        p_dv_2 = tl.make_block_ptr(
            dv, (T * RANK_AB, V), (stride_vo, 1), (i_t * BT * RANK_AB + BT, i_v * BV), 
            (BT, BV), (1, 0)
        ) # [BT, BV]
        
        b_dv_1 = tl.load(p_dv_1, boundary_check=(0, 1)) # [BT, BV]
        b_dv_2 = tl.load(p_dv_2, boundary_check=(0, 1)) # [BT, BV]
        
        b_dw_1 += tl.dot(b_dv_1.to(b_v.dtype), b_h.to(b_v.dtype)) # [BT, BV] @ [BV, BK] -> [BT, BK]
        b_dw_2 += tl.dot(b_dv_2.to(b_v.dtype), b_h.to(b_v.dtype)) # [BT, BV] @ [BV, BK] -> [BT, BK]

    m_k = (i_k * BK + tl.arange(0, BK)) < K
    last_idx = min(i_t * BT + BT, T) - 1
    b_gk_last = tl.load(gk + last_idx * stride_qk + i_k * BK + tl.arange(0, BK), mask=m_k, other=float('-inf'))
    b_dgk_last *= exp(b_gk_last)
    p_k = tl.make_block_ptr(k, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    # b: [B, T * RANK_AB, H, K]
    p_b_1 = tl.make_block_ptr(
        b, (T * RANK_AB, K), (stride_qk, 1), (i_t * BT * RANK_AB, i_k * BK), 
        (BT, BK), (1, 0)
    ) # [BT, BK]
    p_b_2 = tl.make_block_ptr(
        b, (T * RANK_AB, K), (stride_qk, 1), (i_t * BT * RANK_AB + BT, i_k * BK), 
        (BT, BK), (1, 0)
    ) # [BT, BK]
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_b_1 = tl.load(p_b_1, boundary_check=(0, 1))
    b_b_2 = tl.load(p_b_2, boundary_check=(0, 1))

    b_dgk_last += tl.sum(b_k * b_dk, axis=0)
    b_dgk_last += tl.sum(b_b_1 * b_db_1, axis=0)
    b_dgk_last += tl.sum(b_b_2 * b_db_2, axis=0)
    tl.store(dgk_last + tl.arange(0, BK) + i_k * BK, b_dgk_last, mask=m_k)

    # dw: [B, T * RANK_AB, H, K]
    p_dw_1 = tl.make_block_ptr(
        dw, (T * RANK_AB, K), (stride_qk, 1), (i_t * BT * RANK_AB, i_k * BK), 
        (BT, BK), (1, 0)
    )
    p_dw_2 = tl.make_block_ptr(
        dw, (T * RANK_AB, K), (stride_qk, 1), (i_t * BT * RANK_AB + BT, i_k * BK), 
        (BT, BK), (1, 0)
    )
    p_dk = tl.make_block_ptr(dk, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    # db: [B, T * RANK_AB, H, K]
    p_db_1 = tl.make_block_ptr(
        db, (T * RANK_AB, K), (stride_qk, 1), (i_t * BT * RANK_AB, i_k * BK), 
        (BT, BK), (1, 0)
    )
    p_db_2 = tl.make_block_ptr(
        db, (T * RANK_AB, K), (stride_qk, 1), (i_t * BT * RANK_AB + BT, i_k * BK), 
        (BT, BK), (1, 0)
    )
    p_dq = tl.make_block_ptr(dq, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    tl.store(p_dw_1, b_dw_1.to(p_dw_1.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dw_2, b_dw_2.to(p_dw_2.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_db_1, b_db_1.to(p_db_1.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_db_2, b_db_2.to(p_db_2.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8, 16]
        for num_stages in [2, 3, 4]
    ],
    key=['BT', 'BK', 'BV'],
    use_cuda_graph=use_cuda_graph,
)
@triton.jit(do_not_specialize=['T'])
def prepare_wy_repr_bwd_kernel(
    A_ab_inv,
    A_ak,
    ag,
    v,
    dw,
    du,
    dv,
    dv0,
    dag,
    dAak,
    dAab,
    cu_seqlens,
    chunk_indices,
    T,
    RANK_AB: tl.constexpr, 
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BT_AB: tl.constexpr, 
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    bos, eos = i_b * T, i_b * T + T
    # A_ak: [B, T * RANK_AB, H, BT]
    p_Aak_t = tl.make_block_ptr(
        A_ak + (bos * RANK_AB * H + i_h) * BT,  (BT, T * RANK_AB), (1, H * BT), 
        (0, i_t * BT * RANK_AB), (BT, BT * RANK_AB), (0, 1)
    ) # [BT, BT_AB]
    # A_ab: [B, T * RANK_AB, H, BT * RANK_AB]
    p_Aab_inv_t = tl.make_block_ptr(
        A_ab_inv + (bos * RANK_AB * H + i_h) * (BT_AB), (BT_AB, T * RANK_AB), 
        (1, H * BT_AB), (0, i_t * BT_AB), (BT_AB, BT_AB), (0, 1)
    ) # [BT_AB, BT_AB]
    # dAak: [B, T * RANK_AB, H, BT]
    p_dAak = tl.make_block_ptr(
        dAak + (bos * RANK_AB * H + i_h) * BT, (T * RANK_AB, BT), (H * BT, 1), 
        (i_t * BT * RANK_AB, 0), (BT_AB, BT), (1, 0)
    ) # [BT_AB, BT]
    # dAab: [B, T * RANK_AB, H, BT * RANK_AB]
    p_dAab = tl.make_block_ptr(
        dAab + (bos * RANK_AB * H + i_h) * BT_AB, (T * RANK_AB, BT_AB), 
        (H * BT_AB, 1), (i_t * BT_AB, 0), (BT_AB, BT_AB), (1, 0)
    ) # [BT_AB, BT_AB]

    b_A_ab_inv_t = tl.load(p_Aab_inv_t, boundary_check=(0, 1)) # [BT_AB, BT_AB]
    b_A_ak_t = tl.load(p_Aak_t, boundary_check=(0, 1)) # [BT, BT_AB]

    o_kv = tl.arange(0, BT)
    o_ab = tl.interleave(o_kv, o_kv)
    # a_t can only attend to previous (k_1, ..., k_{t - 1})
    b_A_ak_t = tl.where(o_kv[:, None] < o_ab[None, :], b_A_ak_t, 0) # [BT, BT_AB]
    b_A_ab_inv_t = tl.where(o_ab[:, None] <= o_ab[None, :], b_A_ab_inv_t, 0)
    b_A_tmp_t = tl.dot(b_A_ak_t, b_A_ab_inv_t).to(v.dtype.element_ty) # [BT, BT_AB] @ [BT_AB, BT_AB] -> [BT, BT_AB]
    b_dA_tmp = tl.zeros([BT_AB, BT], dtype=tl.float32)

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dv0 = tl.make_block_ptr(dv0 + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        # du: [B, T * RANK_AB, H, V]
        p_du = tl.make_block_ptr(
            du + (bos * RANK_AB * H + i_h) * V, (T * RANK_AB, V), (H * V, 1), 
            (i_t * BT * RANK_AB, i_v * BV), (BT_AB, BV), (1, 0)
        ) # [BT_AB, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1)) # [BT, BV]
        b_du = tl.load(p_du, boundary_check=(0, 1)) # [BT_AB, BV]
        b_dA_tmp += tl.dot(b_du.to(b_v.dtype), tl.trans(b_v)) # [BT_AB, BV] @ [BV, BT] -> [BT_AB, BT]
        b_dv0 = tl.load(p_dv0, boundary_check=(0, 1)) # [BT, BV]
        b_dv = b_dv0 + tl.dot(b_A_tmp_t, b_du) # [BT, BT_AB] @ [BT_AB, BV] -> [BT, BV]
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
    
    m_i_ak = o_ab[:, None] > o_kv[None, :]
    m_i_ab = o_ab[:, None] > o_ab[None, :]
    b_dA_tmp = tl.where(m_i_ak, b_dA_tmp, 0) # [BT_AB, BT]
    b_dA_ak = tl.dot(b_A_ab_inv_t, b_dA_tmp) # [BT_AB, BT_AB] @ [BT_AB, BT] -> [BT_AB, BT]
    b_dA_ak = tl.where(m_i_ak, b_dA_ak, 0) # [BT_AB, BT]
    tl.store(p_dAak, b_dA_ak, boundary_check=(0, 1))
    b_dA_ab_inv = tl.dot(b_dA_tmp, b_A_ak_t) # [BT_AB, BT] @ [BT, BT_AB] -> [BT_AB, BT_AB]

    for i_k in range(tl.cdiv(K, BK)):
        # ag: [B, T * RANK_AB, H, K]
        p_ag = tl.make_block_ptr(
            ag + (bos * RANK_AB * H + i_h) * K, (T * RANK_AB, K), (H * K, 1), 
            (i_t * BT_AB, i_k * BK), (BT_AB, BK), (1, 0)
        ) # [BT_AB, BK]
        # dag: [B, T * RANK_AB, H, K]
        p_dag = tl.make_block_ptr(
            dag + (bos * RANK_AB * H + i_h) * K, (T * RANK_AB, K), (H * K, 1), 
            (i_t * BT_AB, i_k * BK), (BT_AB, BK), (1, 0)
        ) # [BT_AB, BK]
        # dw: [B, T * RANK_AB, H, K]
        p_dw = tl.make_block_ptr(
            dw + (bos * RANK_AB * H + i_h) * K, (T * RANK_AB, K), (H * K, 1), 
            (i_t * BT_AB, i_k * BK), (BT_AB, BK), (1, 0)
        ) # [BT_AB, BK]
        b_ag = tl.load(p_ag, boundary_check=(0, 1)) # [BT_AB, BK]
        b_dw = tl.load(p_dw, boundary_check=(0, 1)) # [BT_AB, BK]
        b_dA_ab_inv += tl.dot(b_dw, tl.trans(b_ag)) # [BT_AB, BK] @ [BK, BT_AB] -> [BT_AB, BT_AB]
        b_dag = tl.dot(b_A_ab_inv_t.to(b_dw.dtype), b_dw) # [BT_AB, BT_AB] @ [BT_AB, BK] -> [BT_AB, BK]
        tl.store(p_dag, b_dag.to(p_dag.dtype.element_ty), boundary_check=(0, 1))
    b_dA_ab_inv = tl.where(o_ab[:, None] >= o_ab[None, :], b_dA_ab_inv, 0)
    b_dA_ab_inv = tl.dot(b_A_ab_inv_t, b_dA_ab_inv)
    b_dA_ab_inv = tl.dot(b_dA_ab_inv, b_A_ab_inv_t)
    b_dA_ab_inv = tl.where(m_i_ab, b_dA_ab_inv, 0)
    tl.store(p_dAab, b_dA_ab_inv, boundary_check=(0, 1))


@triton.jit(do_not_specialize=['T'])
def chunk_hrdplr_bwd_kernel_intra(
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
    dq,
    dk,
    da,
    db,
    dqg,
    dkg,
    dag,
    dbg,
    dgk,
    dgk_offset,
    cu_seqlens,
    chunk_indices,
    scale: tl.constexpr,
    T,
    RANK_AB: tl.constexpr, 
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BT_AB: tl.constexpr, 
    BC: tl.constexpr,
    BC_AB: tl.constexpr, 
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    GATHER_SUPPORTED: tl.constexpr
):
    # grid shape: [NK, NT, B * H]
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    bos, eos = (i_b * T).to(tl.int32), (i_b * T + T).to(tl.int32)
    if i_t * BT >= T:
        return

    # offset calculation
    ge += (bos * H + i_h) * K
    gi += (bos * H + i_h) * K
    q += (bos * H + i_h) * K
    a += (bos * RANK_AB * H + i_h) * K
    b += (bos * RANK_AB * H + i_h) * K
    k += (bos * H + i_h) * K

    dq += (bos * H + i_h) * K
    dk += (bos * H + i_h) * K
    da += (bos * RANK_AB * H + i_h) * K
    db += (bos * RANK_AB * H + i_h) * K

    dqg += (bos * H + i_h) * K
    dag += (bos * RANK_AB * H + i_h) * K
    dkg += (bos * H + i_h) * K
    dbg += (bos * RANK_AB * H + i_h) * K

    dgk += (bos * H + i_h) * K
    dgk_offset += (bos * H + i_h) * K

    dAqk += (bos * H + i_h) * BT
    dAqb += (bos * H + i_h) * BT_AB
    dAak += (bos * RANK_AB * H + i_h) * BT
    dAab += (bos * RANK_AB * H + i_h) * BT_AB

    stride_qk = H * K
    stride_A_k = H * BT
    stride_A_b = H * BT_AB

    p_ge = tl.make_block_ptr(ge, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK), (BC, BK), (1, 0))
    p_gi = tl.make_block_ptr(gi, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK), (BC, BK), (1, 0))
    # [BC, BK]
    b_ge_kv = tl.load(p_ge, boundary_check=(0, 1))
    b_ge_ab = tl.interleave(b_ge_kv.T, b_ge_kv.T).T
    b_gi_kv = tl.load(p_gi, boundary_check=(0, 1))
    b_gi_ab = tl.interleave(b_gi_kv.T, b_gi_kv.T).T
    b_dq = tl.zeros([BC, BK], dtype=tl.float32)
    b_da = tl.zeros([BC_AB, BK], dtype=tl.float32)
    b_dk = tl.zeros([BC, BK], dtype=tl.float32)
    b_db = tl.zeros([BC_AB, BK], dtype=tl.float32)
    # intra chunk gradient calculation
    p_dAqk = tl.make_block_ptr(
        dAqk, (T, BT), (stride_A_k, 1), (i_t * BT, 0), (BC, BC), (1, 0)
    ) # [BC, BC]
    # dAab: [B, T * RANK_AB, H, BT_AB]
    p_dAab = tl.make_block_ptr(
        dAab, (T * RANK_AB, BT_AB), (stride_A_b, 1), (i_t * BT_AB, 0), 
        (BC_AB, BC_AB), (1, 0)
    )
    # dAqb: [B, T, H, BT_AB]
    p_dAqb = tl.make_block_ptr(
        dAqb, (T, BT_AB), (stride_A_b, 1), (i_t * BT, 0), (BC, BC_AB), (1, 0)
    )
    # dAak: [B, T * RANK_AB, H, BT]
    p_dAak = tl.make_block_ptr(
        dAak, (T * RANK_AB, BT), (stride_A_k, 1), (i_t * BT_AB, 0), (BC_AB, BC), 
        (1, 0)
    )
    
    o_i_kv = tl.arange(0, BC)
    o_i_ab = tl.interleave(o_i_kv, o_i_kv)

    range_BT_AB = tl.arange(0, BC_AB)

    p_k = tl.make_block_ptr(k, (T, K), (stride_qk, 1), (i_t*BT, i_k*BK), (BC, BK), (1, 0))
    p_b = tl.make_block_ptr(
        b, (T * RANK_AB, K), (stride_qk, 1), (i_t * BT_AB, i_k * BK), (BC_AB, BK), 
        (1, 0)
    )
    p_a = tl.make_block_ptr(
        a, (T * RANK_AB, K), (stride_qk, 1), (i_t * BT_AB, i_k * BK), (BC_AB, BK), 
        (1, 0)
    )
    p_q = tl.make_block_ptr(q, (T, K), (stride_qk, 1), (i_t*BT, i_k*BK), (BC, BK), (1, 0))
    
    b_k = tl.load(p_k, boundary_check=(0, 1)) # [BC, BK]
    b_b = tl.load(p_b, boundary_check=(0, 1)) # [BC_AB, BK]
    b_q = tl.load(p_q, boundary_check=(0, 1)) # [BC, BK]
    b_a = tl.load(p_a, boundary_check=(0, 1)) # [BC_AB, BK]
    b_dAqk = tl.load(p_dAqk, boundary_check=(0, 1)) # [BC, BC]
    b_dAab = tl.load(p_dAab, boundary_check=(0, 1)) # [BC_AB, BC_AB]
    b_dAqb = tl.load(p_dAqb, boundary_check=(0, 1)) # [BC, BC_AB]
    b_dAak = tl.load(p_dAak, boundary_check=(0, 1)) # [BC_AB, BC]

    # inter chunk gradient calculation
    o_k = i_k * BK + tl.arange(0, BK)
    m_k = o_k < K
    
    # intra chunk gradient calculation
    for j in range(0, min(BC, T - i_t * BT)):
        mask_idx_qk = (o_i_kv == j)
        b_kj = tl.sum(tl.where(mask_idx_qk[:, None], b_k, 0), 0)[None, :] # [1, BK]
        b_gij = tl.sum(tl.where(mask_idx_qk[:, None], b_gi_kv, 0), 0)[None, :] # [1, BK]
        b_gej = tl.sum(tl.where(mask_idx_qk[:, None], b_ge_kv, 0), 0)[None, :] # [1, BK]
        
        # the j-th column of dAqk (the attention of all qs on k_j)
        b_dAqk_j = tl.sum(tl.where(mask_idx_qk[None, :], b_dAqk, 0), 1)[:, None] # [BC, 1]
        # the j-th column of dAak (the attention of all as on k_j)
        b_dAak_j = tl.sum(tl.where(mask_idx_qk[None, :], b_dAak, 0), 1)[:, None] # [BC_AB, 1]
        
        # the j-th row of dAqk (the attention of q_j on all ks)
        b_dA_qk_j = tl.sum(tl.where(mask_idx_qk[:, None], b_dAqk, 0), 0)[:, None] # [BC, 1]
        # the j-th row of dAqb (the attention of q_j on all bs)
        b_dA_qb_j = tl.sum(tl.where(mask_idx_qk[:, None], b_dAqb, 0), 0)[:, None] # [BC_AB, 1]
            
        b_qj = tl.sum(tl.where(mask_idx_qk[:, None], b_q, 0), 0)[None, :] # [1, BK]

        # row masks
        # part 1
        m_e_a = o_i_ab[:, None] > j
        m_i_q = o_i_kv[:, None] >= j

        tmp_i_q = exp(b_gi_kv - b_gij) # [BC, BK]
        tmp_e_q = exp(b_ge_kv - b_gij) # [BC, BK]
        tmp_e_a = tl.interleave(tmp_e_q.T, tmp_e_q.T).T # [BC_AB, BK]

        b_dq += tl.where(m_i_q, b_dAqk_j * b_kj * tmp_i_q, 0.) # [BC, 1] * [1, BK] * [BC, BK]
        b_da += tl.where(m_e_a, b_dAak_j * b_kj * tmp_e_a, 0.) # [BC_AB, 1] * [1, BK] * [BC_AB, BK]

        # part 2
        m_i_k = o_i_kv[:, None] <= j
        m_i_b = o_i_ab[:, None] <= j
        m_e_k = o_i_kv[:, None] < j
        m_e_b = o_i_ab[:, None] < j

        tmp_i_k = exp(b_gij - b_gi_kv) # [BC, BK]
        tmp_i_b = tl.interleave(tmp_i_k.T, tmp_i_k.T).T # [BC_AB, BK]
        tmp_e_k = exp(b_gej - b_gi_kv) # [BC, BK]
        tmp_e_b = tl.interleave(tmp_e_k.T, tmp_e_k.T).T # [BC_AB, BK]

        b_dk += tl.where(m_i_k, b_dA_qk_j * b_qj * tmp_i_k, 0.) # [BC, 1] * [1, BK] * [BC, BK]
        b_db += tl.where(m_i_b, b_dA_qb_j * b_qj * tmp_i_b, 0.) # [BC_AB, 1] * [1, BK] * [BC_AB, BK]

        for i_r in range(RANK_AB):
            
            mask_idx_ab = (range_BT_AB == j * RANK_AB + i_r)
            b_aj = tl.sum(tl.where(mask_idx_ab[:, None], b_a, 0), 0)[None, :] # [1, BK]
            b_bj = tl.sum(tl.where(mask_idx_ab[:, None], b_b, 0), 0)[None, :] # [1, BK]
            
            # the (j * RANK_AB + i_r)-column of dAab (the attention of all as on b_{j, i_r})
            b_dAab_j = tl.sum(tl.where(mask_idx_ab[None, :], b_dAab, 0), 1)[:, None] # [BC_AB, 1]
            # the (j * RANK_AB + i_r)-column of dAqb (the attention of all qs on b_{j, i_r})
            b_dAqb_j = tl.sum(tl.where(mask_idx_ab[None, :], b_dAqb, 0), 1)[:, None] # [BC, 1]
            
            # the (j * RANK_AB + i_r)-th row of dAab (the attention of a_j on all bs)
            b_dA_ab_j = tl.sum(tl.where(mask_idx_ab[:, None], b_dAab, 0), 0)[:, None] # [BC_AB, ]
            # the (j * RANK_AB + i_r)-th row of dAak (the attention of a_j on all ks)
            b_dA_ak_j = tl.sum(tl.where(mask_idx_ab[:, None], b_dAak, 0), 0)[:, None] # [BC, ]

            # part 1
            b_dq += tl.where(m_i_q, b_dAqb_j * b_bj * tmp_i_q, 0.) # [BC, 1] * [1, BK] * [BC, BK]
            b_da += tl.where(m_e_a, b_dAab_j * b_bj * tmp_e_a, 0.) # [BC_AB, 1] * [1, BK] * [BC_AB, BK]

            # part 2
            b_dk += tl.where(m_e_k, b_dA_ak_j * b_aj * tmp_e_k, 0.) # [BC, 1] * [1, BK] * [BC, BK]
            b_db += tl.where(m_e_b, b_dA_ab_j * b_aj * tmp_e_b, 0.) # [BC_AB, 1] * [1, BK] * [BC_AB, BK]
    
    p_dq = tl.make_block_ptr(dq, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK), (BC, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK), (BC, BK), (1, 0))
    p_da = tl.make_block_ptr(
        da, (T * RANK_AB, K), (stride_qk, 1), 
        (i_t * BT_AB, i_k * BK), (BC_AB, BK), (1, 0)
    )
    p_db = tl.make_block_ptr(
        db, (T * RANK_AB, K), (stride_qk, 1), 
        (i_t * BT_AB, i_k * BK), (BC_AB, BK), (1, 0)
    )
    p_dgk = tl.make_block_ptr(dgk, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK), (BC, BK), (1, 0))
    p_dgk_offset = tl.make_block_ptr(dgk_offset, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK), (BC, BK), (1, 0))
    p_dqg = tl.make_block_ptr(dqg, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK), (BC, BK), (1, 0))
    p_dkg = tl.make_block_ptr(dkg, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK), (BC, BK), (1, 0))
    p_dag = tl.make_block_ptr(
        dag, (T * RANK_AB, K), (stride_qk, 1), (i_t * BT_AB, i_k * BK), (BC_AB, BK),
        (1, 0)
    )
    p_dbg = tl.make_block_ptr(
        dbg, (T * RANK_AB, K), (stride_qk, 1), (i_t * BT_AB, i_k * BK), (BC_AB, BK), 
        (1, 0)
    )
    p_gn = gi + (min(i_t * BT + BT, T) - 1) * stride_qk + o_k
    p_gn = tl.max_contiguous(tl.multiple_of(p_gn, BK), BK)
    b_gn = tl.load(p_gn, mask=m_k, other=0)
    b_da += tl.load(p_dag, boundary_check=(0, 1)) * exp(b_ge_ab)
    b_dq += tl.load(p_dqg, boundary_check=(0, 1)) * exp(b_gi_kv) * scale
    
    tmp_k = exp(b_gn[None, :] - b_gi_kv)
    tmp_b = exp(b_gn[None, :] - b_gi_ab)

    b_dk += tl.load(p_dkg, boundary_check=(0, 1)).to(tl.float32) * tmp_k
    b_db += tl.load(p_dbg, boundary_check=(0, 1)).to(tl.float32) * tmp_b

    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_da, b_da.to(p_da.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_db, b_db.to(p_db.dtype.element_ty), boundary_check=(0, 1))

    b_dgk = (
        b_dq * b_q 
        + tl.sum(
            tl.reshape(b_da * b_a, (BC, RANK_AB, BK)), axis=1
        ) 
        - b_dk * b_k 
        - tl.sum(
            tl.reshape(b_db * b_b, (BC, RANK_AB, BK)), axis=1
        )
    ).to(tl.float32)
    b_dgk_offset = tl.sum(
        tl.reshape(b_da * b_a, (BC, RANK_AB, BK)), axis=1
    )
    
    tl.store(p_dgk, b_dgk.to(p_dgk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dgk_offset, b_dgk_offset.to(p_dgk_offset.dtype.element_ty), boundary_check=(0, 1))

@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS_AUTOTUNE
        for num_stages in [2, 3, 4]
        for BK in [32, 64]
    ],
    key=['BK', 'BT', 'K'],
    use_cuda_graph=use_cuda_graph,
)
@triton.jit(do_not_specialize=['T'])
def chunk_dplr_bwd_dgk_kernel(
    dgk,
    dgk_offset,
    dgk_last,
    dgk_output,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    This kernel is directly borrowed from:
    https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/generalized_delta_rule/dplr/chunk_A_bwd.py
    """
    i_t, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = (i_b * NT + i_t).to(tl.int32)
        bos, eos = (i_b * T).to(tl.int32), (i_b * T + T).to(tl.int32)

    stride_qk = H * K
    dgk += (bos * H + i_h) * K
    dgk_offset += (bos * H + i_h) * K
    dgk_last += (i_tg * H + i_h) * K
    dgk_output += (bos * H + i_h) * K
    p_dgk_last = dgk_last + tl.arange(0, BK) + i_k * BK
    m_k = tl.arange(0, BK) + i_k * BK < K
    b_dgk_last = tl.load(p_dgk_last, mask=m_k, other=0)
    p_dgk_offset = tl.make_block_ptr(dgk_offset, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dgk = tl.make_block_ptr(dgk, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    b_dgk = tl.load(p_dgk, boundary_check=(0, 1))
    b_dgk_offset = tl.load(p_dgk_offset, boundary_check=(0, 1))
    # m_inv_cumsum = (tl.arange(0, BT)[:, None] <= tl.arange(0, BT)[None, :]).to(tl.float32)
    # b_dgk_cumsum = tl.dot(m_inv_cumsum, b_dgk, allow_tf32=False)
    b_dgk_cumsum = tl.cumsum(b_dgk, 0, reverse=True)
    b_dgk_cumsum += b_dgk_last[None, :]
    b_dgk_cumsum -= b_dgk_offset
    p_dgk_output = tl.make_block_ptr(dgk_output, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    tl.store(p_dgk_output, b_dgk_cumsum.to(p_dgk_output.dtype.element_ty), boundary_check=(0, 1))