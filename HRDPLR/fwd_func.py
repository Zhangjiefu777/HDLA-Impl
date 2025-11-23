import torch
import triton

from typing import Optional, Tuple

from .fwd_triton import (
    chunk_dplr_fwd_A_kernel_intra_sub_intra_rab_generalized, 
    chunk_dplr_fwd_A_kernel_intra_sub_inter_rab_generalized, 
    fwd_prepare_wy_repr_kernel_chunk32, 
    fwd_wu_kernel, 
    chunk_dplr_fwd_kernel_h, 
    chunk_dplr_fwd_kernel_o, 
)

def chunk_fwd_intra_dplr_fn(
    q: torch.Tensor, # [B, T, H, K] / [B, T, H, K]
    k: torch.Tensor, # [B, T, H, K] / [B, T, H, K]
    a: torch.Tensor, # [B, T, H, K] / [B, T, H, K]
    b: torch.Tensor, # [B, T, H, K] / [B, T, H, K]
    gi: torch.Tensor, # [B, T, H, K] / [B, T, H, K]
    ge: torch.Tensor, 
    RANK_AB: int, 
    scale: float, 
    chunk_size: int = 16, 
    head_first: bool = False, 
):
    assert head_first == False
    B, T, H, K = k.shape

    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    NT = triton.cdiv(T, BT)
    
    BC = min(16, BT)
    NC = triton.cdiv(BT, BC)

    device = q.device
    dtype = q.dtype

    Aqk = torch.empty(size=(B, T, H, BT), device=device, dtype=dtype)
    Aqb = torch.empty(size=(B, T, H, BT * RANK_AB), device=device, dtype=dtype)
    Aak = torch.empty(size=(B, T * RANK_AB, H, BT), device=device, dtype=torch.float)
    Aab = torch.empty(size=(B, T * RANK_AB, H, BT * RANK_AB), device=device, dtype=torch.float)

    grid = (NT, NC * NC, B * H)
    chunk_dplr_fwd_A_kernel_intra_sub_inter_rab_generalized[grid](
        q, # [B, T, H, K]
        k, # [B, T, H, K]
        a, # [B, T, H, K]
        b, # [B, T, H, K]
        gi, # [B, T, H, K]
        ge, # [B, T, H, K]
        # intermediate results
        Aqk, # [B, T, H, BT]
        Aqb, # [B, T, H, BT]
        Aab, # [B, T, H, BT]
        Aak, # [B, T, H, BT]
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

    BK = triton.next_power_of_2(K)
    
    qg = torch.empty_like(q)
    kg = torch.empty_like(k, dtype=q.dtype)
    ag = torch.empty_like(a, dtype=q.dtype)
    bg = torch.empty_like(b, dtype=q.dtype)

    grid = (NT, NC, B * H)
    chunk_dplr_fwd_A_kernel_intra_sub_intra_rab_generalized[grid](
        q, # [B, T, H, K] / [B, T, H, K]
        k, # [B, T, H, K] / [B, T, H, K]
        a, # [B, T * RANK_AB, H, K] / [B, T * RANK_AB, H, K]
        b, # [B, T * RANK_AB, H, K] / [B, T * RANK_AB, H, K]
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
        RANK_AB, 
        scale,
        T,
        H,
        K,
        BT, # BT: coarse-grained chunk size along T dim
        BC, # BC: fine-grained chunk size along T dim
        BC * RANK_AB, # BC * RANK_AB
        BK,
        NC,
    )
    return Aqk, Aqb, Aak, Aab, qg, kg, ag, bg

def fwd_wu(
    ag: torch.Tensor,
    v: torch.Tensor,
    A_ak: torch.Tensor,
    A_ab_inv: torch.Tensor,
    head_first: bool,
    chunk_size: int, 
    RANK_AB: int, 
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert head_first == False

    B, T_ab, H, K, V = *ag.shape, v.shape[-1]
    T = T_ab // RANK_AB

    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))
    NT = triton.cdiv(T, BT)

    BK = min(triton.next_power_of_2(K), 64)
    BV = min(triton.next_power_of_2(V), 64)

    u = torch.empty(
        size=(B, T * RANK_AB, H, V), device=ag.device, dtype=ag.dtype, 
    )
    w = torch.empty_like(ag)

    grid = (NT, B * H)
    fwd_wu_kernel[grid](
        u, # [B, T * RANK_AB, H, K]
        w, # [B, T * RANK_AB, H, V]
        ag, # [B, T, H, K]
        v, # [B, T, H, V] / [B, T, H, V]
        A_ab_inv, # [B, T * RANK_AB, H, BT * RANK_AB]
        A_ak, # [B, T * RANK_AB, H, BT]
        T,
        H,
        K,
        V,
        BT,
        RANK_AB, 
        BT * RANK_AB, 
        # BK,
        # BV,
    )

    return w, u

def fwd_prepare_wy_repr(
    ag: torch.Tensor, # [B, T * RANK_AB, H, K]
    v: torch.Tensor, # [B, T, H, V]
    A_ak: torch.Tensor, # [B, T * RANK_AB, H, BT]
    A_ab: torch.Tensor, # [B, T * RANK_AB, H, BT]
    chunk_size: int, 
    RANK_AB: int, 
    head_first: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert head_first == False
    B, T_ab, H, _ = ag.shape
    T = T_ab // RANK_AB

    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))
    NT = triton.cdiv(T, BT)

    BC = min(BT, 16)
    
    fwd_fn = fwd_prepare_wy_repr_kernel_chunk32

    A_ab_inv = torch.empty_like(A_ab)
    
    grid = (NT, B * H)
    fwd_fn[grid](
        A_ab, # [B, T * RANK_AB, H, BT * RANK_AB]
        A_ab_inv, # [B, T * RANK_AB, H, BT * RANK_AB]
        B, 
        H, 
        T, 
        BT, 
        RANK_AB, 
    )

    w, u = fwd_wu(
        ag,
        v,
        A_ak,
        A_ab_inv,
        head_first,
        chunk_size, 
        RANK_AB, 
    )

    return w, u, A_ab_inv

def chunk_dplr_fwd_h(
    kg: torch.Tensor, # [B, T, H, K] 
    v: torch.Tensor, # [B, T, H, V]
    w: torch.Tensor, # [B, T * RANK_AB, H, K]
    u: torch.Tensor, # [B, T * RANK_AB, H, V]
    bg: torch.Tensor, # [B, T * RANK_AB, H, K]
    gk: torch.Tensor, # [B, T, H, K]
    RANK_AB: int, 
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    head_first: bool = False,
    chunk_size: int = 64
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert head_first == False

    B, T, H, K, V = *kg.shape, u.shape[-1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))

    N, NT = B, triton.cdiv(T, BT)
    BK = triton.next_power_of_2(K)
    assert BK <= 256, "current kernel does not support head dimension larger than 256."
    # H100 can have larger block size
    if torch.cuda.get_device_capability()[0] >= 9:
        BV = 64
        BC = 64 if K <= 128 else 32
    else:
        BV = 32
        BC = 32
    BC = min(BT, BC)
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    assert NK == 1, 'NK > 1 is not supported because it involves time-consuming synchronization'

    h = kg.new_empty(B, NT, H, K, V)
    final_state = kg.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None
    v_new = torch.empty_like(u)

    grid = (NK, NV, N * H)
    chunk_dplr_fwd_kernel_h[grid](
        kg, # [B, T, H, K]
        v, # [B, T, H, V]        
        w, # [B, T * RANK_AB, H, K]
        bg, # [B, T * RANK_AB, H, K]
        u, # [B, T * RANK_AB, H, K]
        v_new, # [B, T * RANK_AB, H, V]
        gk, # [B, T, H, K]
        h, # [B, NT, H, K, V]
        initial_state, # [B, H, K, V]
        final_state, # [B, H, K, V]
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

    return h, v_new, final_state

def chunk_dplr_fwd_o(
    qg: torch.Tensor,
    v: torch.Tensor,
    v_new: torch.Tensor,
    A_qk: torch.Tensor,
    A_qb: torch.Tensor,
    h: torch.Tensor,
    RANK_AB: int, 
    head_first: bool = False,
    chunk_size: int = 64, 
):
    assert head_first == False
    B, T, H, K, V = *qg.shape, v.shape[-1]

    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    NT = triton.cdiv(T, BT)

    o = torch.empty_like(v)

    def grid(meta): return (triton.cdiv(V, meta['BV']), NT, B * H)
    chunk_dplr_fwd_kernel_o[grid](
        qg, # [B, T, H, K]
        v,
        v_new,
        A_qk,
        A_qb,
        h,
        o,
        T,
        H,
        K,
        V,
        BT, 
        RANK_AB, 
        BT * RANK_AB, 
    )

    return o