from typing import Optional, Tuple

import torch
import triton
from fla.utils import check_shared_mem, is_gather_supported
from fla.ops.utils import prepare_chunk_indices

from .bwd_triton import (
    chunk_dplr_bwd_kernel_dAu, 
    chunk_dplr_bwd_kernel_dhu, 
    chunk_dplr_bwd_kernel_dv, 
    chunk_dplr_bwd_o_kernel, 
    prepare_wy_repr_bwd_kernel, 
    chunk_hrdplr_bwd_kernel_intra, 
    chunk_dplr_bwd_dgk_kernel, 
)

def chunk_dplr_bwd_dAu(
    v: torch.Tensor, 
    v_new: torch.Tensor, 
    do: torch.Tensor,
    A_qb: torch.Tensor,
    scale: float,
    RANK_AB: int, 
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64
):
    B, T, H, V = v.shape
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))

    NT = triton.cdiv(T, BT)
    if check_shared_mem('ampere'):  # A100
        BV = min(triton.next_power_of_2(V), 128)
    elif check_shared_mem('ada'):  # 4090
        BV = min(triton.next_power_of_2(V), 64)
    else:
        BV = min(triton.next_power_of_2(V), 32)

    dA_qk = torch.empty(B, T, H, BT, dtype=torch.float, device=v.device)
    dA_qb = torch.empty(B, T, H, BT * RANK_AB, dtype=torch.float, device=v.device)
    dv_new = torch.empty_like(v_new)

    grid = (NT, B * H)
    chunk_dplr_bwd_kernel_dAu[grid](
        v, # [B, T, H, V]
        do, # [B, T, H, V]
        v_new, # [B, T * RANK_AB, H, V]
        A_qb, # [B, T, H, BT * RANK_AB]
        dA_qk, # [B, T, H, BT]
        dA_qb, # [B, T, H, BT * RANK_AB]
        dv_new, # [B, T * RANK_AB, H, V]
        scale,
        T,
        RANK_AB, 
        BT * RANK_AB, 
        H,
        V,
        BT,
        BV,
    )
    return dv_new, dA_qk, dA_qb

def chunk_dplr_bwd_dhu(
    qg: torch.Tensor,
    bg: torch.Tensor,
    w: torch.Tensor,
    gk: torch.Tensor,
    h0: torch.Tensor,
    dht: Optional[torch.Tensor],
    do: torch.Tensor,
    dv: torch.Tensor,
    RANK_AB: int, 
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *qg.shape, do.shape[-1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))
    BK = triton.next_power_of_2(K)

    assert BK <= 256, "current kernel does not support head dimension being larger than 256."
    # H100
    if check_shared_mem('hopper', qg.device.index):
        BV = 64
        BC = 64 if K <= 128 else 32
    elif check_shared_mem('ampere', qg.device.index):  # A100
        BV = 32
        BC = 32
    else:  # Etc: 4090
        BV = 16
        BC = 16

    N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    dh = qg.new_empty(B, NT, H, K, V)
    dh0 = torch.empty_like(h0, dtype=torch.float32) if h0 is not None else None
    dv2 = torch.zeros_like(dv)

    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)

    grid = (NK, NV, N * H)
    chunk_dplr_bwd_kernel_dhu[grid](
        qg=qg, # [B, T, H, K]
        bg=bg, # [B, T, H, K]
        w=w, # [B, T * RANK_AB, H, K]
        gk=gk, # [B, T, H, K]
        dht=dht, # [B, H, K, V]
        dh0=dh0, # [B, H, K, V]
        do=do, # [B, H, ]
        dh=dh, # [B, NT, H, K, V]
        dv=dv, # (dv_new_intra) [B, T * RANK_AB, H, V]
        dv2=dv2, # (dv_new) [B, T * RANK_AB, H, V]
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BC=BC,
        BK=BK,
        BV=BV,
        RANK_AB=RANK_AB, 
    )
    return dh, dh0, dv2

def chunk_dplr_bwd_dv(
    A_qk: torch.Tensor,
    kg: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64
) -> torch.Tensor:
    B, T, H, K, V = *kg.shape, do.shape[-1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))

    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    dv = torch.empty_like(do)
    def grid(meta): return (triton.cdiv(V, meta['BV']), NT, B * H)
    chunk_dplr_bwd_kernel_dv[grid](
        A_qk=A_qk, # [B, T, H, BT]
        kg=kg, # [B, T, H, K]
        do=do, # [B, T, H, V]
        dv=dv, # [B, T, H, V]
        dh=dh, # [B, NT, H, K, V]
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
    )
    return dv

def chunk_dplr_bwd_o(
    k: torch.Tensor,
    b: torch.Tensor, 
    v: torch.Tensor,
    v_new: torch.Tensor,
    gk: torch.Tensor,
    do: torch.Tensor,
    h: torch.Tensor,
    dh: torch.Tensor,
    dv: torch.Tensor,
    w: torch.Tensor,
    RANK_AB: int, 
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
    scale: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    B, T, H, K, V = *k.shape, v.shape[-1]

    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    BK = min(triton.next_power_of_2(K), 64) if check_shared_mem() else min(triton.next_power_of_2(K), 32)
    BV = min(triton.next_power_of_2(V), 64) if check_shared_mem() else min(triton.next_power_of_2(K), 32)
    NK = triton.cdiv(K, BK)
    dq = torch.empty_like(k)
    dk = torch.empty_like(k)
    dw = torch.empty_like(w)
    db = torch.empty_like(b)
    grid = (NK, NT, B * H)

    dgk_last = torch.empty(B, NT, H, K, dtype=torch.float, device=w.device)
    chunk_dplr_bwd_o_kernel[grid](
        v=v, # [B, T, H, V]
        v_new=v_new, # [B, T * RANK_AB, H, V]
        h=h, # [B, NT, H, K, V]
        do=do, # [B, T, H, K]
        dh=dh, # [B, NT, H, K, V]
        dk=dk,
        db=db,
        w=w,
        dq=dq,
        dv=dv, # (dv_new) [B, T * RANK_AB, H, K]
        dw=dw,
        gk=gk,
        dgk_last=dgk_last,
        k=k,
        b=b,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        V=V,
        RANK_AB=RANK_AB, 
        BT=BT,
        BT_AB=BT * RANK_AB, 
        BK=BK,
        BV=BV,
    )
    return dq, dk, dw, db, dgk_last

def chunk_dplr_bwd_wy(
    A_ab_inv: torch.Tensor,
    A_ak: torch.Tensor,
    v: torch.Tensor,
    ag: torch.Tensor,
    dw: torch.Tensor,
    du: torch.Tensor,
    dv0: torch.Tensor,
    RANK_AB: int, 
    cu_seqlens: Optional[torch.LongTensor],
    chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    A_ab_inv, A_ak, v, ag, dw, du = map(lambda x: x.contiguous(), [A_ab_inv, A_ak, v, ag, dw, du])
    B, T_ab, H, K, V = *dw.shape, du.shape[-1]

    T = T_ab // RANK_AB

    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))

    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BK = min(triton.next_power_of_2(K), 64)
    BV = min(triton.next_power_of_2(V), 64) if check_shared_mem() else min(triton.next_power_of_2(V), 32)

    dA_ab = torch.empty_like(A_ab_inv, dtype=torch.float)
    dA_ak = torch.empty_like(A_ak, dtype=torch.float)
    dv = torch.empty_like(v)
    dag = torch.empty_like(ag)

    grid = [NT, B * H]
    prepare_wy_repr_bwd_kernel[grid](
        A_ab_inv=A_ab_inv,
        A_ak=A_ak,
        ag=ag,
        v=v,
        dw=dw,
        du=du,
        dv=dv,
        dv0=dv0,
        dag=dag,
        dAak=dA_ak,
        dAab=dA_ab,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        RANK_AB=RANK_AB, 
        H=H,
        K=K,
        V=V,
        BT=BT,
        BT_AB=BT * RANK_AB, 
        BK=BK,
        BV=BV,
    )
    return dA_ab, dA_ak, dv, dag

def chunk_dplr_bwd_dqk_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    gi: torch.Tensor,
    ge: torch.Tensor,
    dAqk: torch.Tensor,
    dAqb: torch.Tensor,
    dAak: torch.Tensor,
    dAab: torch.Tensor,
    dqg: torch.Tensor,
    dkg: torch.Tensor,
    dag: torch.Tensor,
    dbg: torch.Tensor,
    dgk_last: torch.Tensor,
    RANK_AB: int, 
    scale: float = 1.0,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
):
    B, T, H, K = q.shape
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    BK = min(64, triton.next_power_of_2(K)) if check_shared_mem() else min(32, triton.next_power_of_2(K))

    BC = 16 # TODO

    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NK = triton.cdiv(K, BK)

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    da = torch.empty_like(a)
    db = torch.empty_like(b)
    dgk = torch.empty_like(gi, dtype=torch.float)
    dgk_offset = torch.empty_like(gi, dtype=torch.float)

    grid = (NK, NT, B * H)
    chunk_hrdplr_bwd_kernel_intra[grid](
        q=q, 
        k=k, 
        a=a, 
        b=b, 
        gi=gi, 
        ge=ge, 
        dAqk=dAqk,
        dAqb=dAqb,
        dAak=dAak,
        dAab=dAab,
        dq=dq,
        dk=dk,
        da=da,
        db=db,
        dqg=dqg,
        dkg=dkg,
        dag=dag,
        dbg=dbg,
        dgk=dgk,
        dgk_offset=dgk_offset,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        RANK_AB=RANK_AB, 
        H=H,
        K=K,
        BT=BT,
        BT_AB=BT * RANK_AB, 
        BC=BC,
        BC_AB=BC * RANK_AB, 
        BK=BK,
        IS_VARLEN=False, 
        GATHER_SUPPORTED=is_gather_supported, 
    )

    dgk_output = torch.empty_like(dgk)
    def grid(meta): return (NT, triton.cdiv(K, meta['BK']), B * H)
    chunk_dplr_bwd_dgk_kernel[grid](
        dgk=dgk,
        dgk_offset=dgk_offset,
        dgk_last=dgk_last,
        dgk_output=dgk_output,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        BT=BT,
    )
    return dq, dk, da, db, dgk_output