from typing import Optional, Tuple

import torch
import triton

from fla.ops.rwkv6.chunk import chunk_rwkv6_fwd_cumsum

from .utils import autocast_custom_bwd, autocast_custom_fwd, contiguous
from .fwd_func import (
    chunk_fwd_intra_dplr_fn, 
    fwd_prepare_wy_repr, 
    chunk_dplr_fwd_h, 
    chunk_dplr_fwd_o, 
)

from .bwd_func import (
    chunk_dplr_bwd_dAu, 
    chunk_dplr_bwd_dhu, 
    chunk_dplr_bwd_dv, 
    chunk_dplr_bwd_o, 
    chunk_dplr_bwd_wy, 
    chunk_dplr_bwd_dqk_intra, 
)

def chunk_hrdplr_fwd(
    q: torch.Tensor, # [B, T, H, K]
    k: torch.Tensor, # [B, T, H, K]
    v: torch.Tensor, # [B, T, H, K]
    a: torch.Tensor, # [B, T * RANK_AB, H, K]
    b: torch.Tensor, # [B, T * RANK_AB, H, K]
    gk: torch.Tensor, # [B, T, H, K]
    scale: float, 
    initial_state: torch.Tensor, # [B, T, H, K]
    output_final_state: bool,

    RANK_AB: int, 
    head_first: bool, 
    chunk_size: int, 
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    T = q.shape[1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))
    
    gi_kv, ge_kv = chunk_rwkv6_fwd_cumsum(
        gk, BT, cu_seqlens=None, 
    )

    Aqk, Aqb, Aak, Aab, qg, kg, ag, bg = chunk_fwd_intra_dplr_fn(
        q=q, # [B, T, H, K] / [B, T, H, K]
        k=k, # [B, T, H, K] / [B, T, H, K]
        a=a, # [B, T, H, K] / [B, T, H, K]
        b=b, # [B, T, H, K] / [B, T, H, K]
        gi=gi_kv, # [B, T, H, K] / [B, T, H, K]
        ge=ge_kv, 
        RANK_AB=RANK_AB, 
        scale=scale, 
        chunk_size=chunk_size, 
        head_first=head_first, 
    )

    w, u, A_ab_inv = fwd_prepare_wy_repr(
        ag, # [B, T * RANK_AB, H, K]
        v, # [B, T, H, V]
        Aak, # [B, T * RANK_AB, H, BT]
        Aab, # [B, T * RANK_AB, H, BT]
        chunk_size, 
        RANK_AB, 
        head_first,
    )

    h, v_new, final_state = chunk_dplr_fwd_h(
        kg, 
        v, 
        w, 
        u,
        bg,
        gi_kv,
        RANK_AB, 
        initial_state,
        output_final_state,
        head_first,
        chunk_size, 
    )

    
    o = chunk_dplr_fwd_o(
        qg,
        v,
        v_new,
        Aqk,
        Aqb,
        h,
        RANK_AB, 
        head_first,
        chunk_size, 
    )

    # return o, final_state, w, u, Aqk, Aqb, Aak, Aab, A_ab_inv, kg, bg#, h, final_state, v_new
    return o, final_state

class ChunkDPLRDeltaRuleFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(
        ctx, 
        q: torch.Tensor, # [B, T, H, K]
        k: torch.Tensor, # [B, T, H, K]
        v: torch.Tensor, # [B, T, H, K]
        a: torch.Tensor, # [B, T * RANK_AB, H, K]
        b: torch.Tensor, # [B, T * RANK_AB, H, K]
        gk: torch.Tensor, # [B, T, H, K]
        scale: float, 
        initial_state: torch.Tensor, # [B, T, H, K]
        output_final_state: bool,

        RANK_AB: int, 
        head_first: bool, 
        chunk_size: int, 
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        o, final_state = chunk_hrdplr_fwd(
            q=q, # [B, T, H, K]
            k=k, # [B, T, H, K]
            v=v, # [B, T, H, K]
            a=a, # [B, T * RANK_AB, H, K]
            b=b, # [B, T * RANK_AB, H, K]
            gk=gk, # [B, T, H, K]
            scale=scale, 
            initial_state=initial_state, # [B, T, H, K]
            output_final_state=output_final_state,
            RANK_AB=RANK_AB, 
            head_first=head_first, 
            chunk_size=chunk_size, 
        )

        ctx.save_for_backward(q, k, v, a, b, gk, initial_state)
        ctx.head_first = head_first
        ctx.scale = scale
        ctx.chunk_size = chunk_size
        ctx.output_final_state = output_final_state
        ctx.RANK_AB = RANK_AB

        return o.to(q.dtype), final_state
    
    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(
        ctx,
        do: torch.Tensor,
        dht: torch.Tensor
    ):
        q, k, v, a, b, gk, initial_state = ctx.saved_tensors
        BT = ctx.chunk_size
        RANK_AB = ctx.RANK_AB
        output_final_state = ctx.output_final_state
        head_first = ctx.head_first
        scale = ctx.scale

        gi_kv, ge_kv = chunk_rwkv6_fwd_cumsum(
            gk, BT, cu_seqlens=None, # [B, T, H, K]
        )
        Aqk, Aqb, Aak, Aab, qg, kg, ag, bg = chunk_fwd_intra_dplr_fn(
            q=q, # [B, T, H, K] / [B, T, H, K]
            k=k, # [B, T, H, K] / [B, T, H, K]
            a=a, # [B, T, H, K] / [B, T, H, K]
            b=b, # [B, T, H, K] / [B, T, H, K]
            gi=gi_kv, # [B, T, H, K] / [B, T, H, K]
            ge=ge_kv, 
            RANK_AB=RANK_AB, 
            scale=scale, 
            chunk_size=BT, 
            head_first=head_first, 
        )
        w, u, A_ab_inv = fwd_prepare_wy_repr(
            ag, # [B, T * RANK_AB, H, K]
            v, # [B, T, H, V]
            Aak, # [B, T * RANK_AB, H, BT]
            Aab, # [B, T * RANK_AB, H, BT]
            BT, 
            RANK_AB, 
            head_first,
        )
        del Aab
        h, v_new, final_state = chunk_dplr_fwd_h(
            kg, 
            v, 
            w, 
            u,
            bg,
            gi_kv,
            RANK_AB, 
            initial_state,
            output_final_state,
            head_first,
            BT, 
        )
        del u
        dv_new_intra, dA_qk, dA_qb = chunk_dplr_bwd_dAu(
            v=v, 
            v_new=v_new, 
            do=do,
            A_qb=Aqb,
            scale=scale,
            RANK_AB=RANK_AB, 
            cu_seqlens=None,
            chunk_size=BT, 
        )
        dh, dh0, dv_new = chunk_dplr_bwd_dhu(
            qg=qg,
            bg=bg,
            w=w,
            gk=gi_kv,
            h0=initial_state,
            dht=dht,
            do=do,
            dv=dv_new_intra,
            RANK_AB=RANK_AB, 
            cu_seqlens=None,
            chunk_size=BT
        )
        dv = chunk_dplr_bwd_dv(
            A_qk=Aqk,
            kg=kg,
            do=do,
            dh=dh,
            cu_seqlens=None,
            chunk_size=BT
        )
        del Aqk
        dqg, dkg, dw, dbg, dgk_last = chunk_dplr_bwd_o(
            k=kg,
            b=bg, # []
            v=v,
            v_new=v_new,
            gk=gi_kv,
            do=do,
            h=h,
            dh=dh,
            dv=dv_new,
            w=w,
            RANK_AB=RANK_AB, 
            cu_seqlens=None,
            chunk_size=BT,
            scale=scale,
        )
        del v_new
        dA_ab, dA_ak, dv, dag = chunk_dplr_bwd_wy(
            A_ab_inv=A_ab_inv,
            A_ak=Aak,
            v=v,
            ag=ag,
            dw=dw,
            du=dv_new,
            dv0=dv,
            RANK_AB=RANK_AB, 
            cu_seqlens=None,
            chunk_size=BT,
        )
        del Aak
        dq, dk, da, db, dgk = chunk_dplr_bwd_dqk_intra(
            q=q,
            k=k,
            a=a,
            b=b,
            gi=gi_kv,
            ge=ge_kv,
            dAqk=dA_qk,
            dAqb=dA_qb,
            dAak=dA_ak,
            dAab=dA_ab,
            dqg=dqg,
            dkg=dkg,
            dag=dag,
            dbg=dbg,
            dgk_last=dgk_last,
            RANK_AB=RANK_AB, 
            scale=scale,
            cu_seqlens=None,
            chunk_size=BT,
        )

        return dq.to(q), dk.to(k), dv.to(v), da.to(a), db.to(b), dgk.to(gk), None, dh0, None, None, None, None

@torch.compiler.disable
def chunk_hrdplr_delta_rule(
    q: torch.Tensor, # [B, T, H, K]
    k: torch.Tensor, # [B, T, H, K]
    v: torch.Tensor, # [B, T, H, V]
    a: torch.Tensor, # [B, T * RANK_AB, H, K]
    b: torch.Tensor, # [B, T * RANK_AB, H, K]
    gk: torch.Tensor, # [B, T, H, K]
    RANK_AB: int, 
    scale: Optional[float] = None, # equivalent to apply row-wise on q
    initial_state: Optional[torch.Tensor] = None, # [B, H, K, V]
    output_final_state: bool = False, 
    head_first: bool = False, 
    chunk_size: int = 32, 
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert q.dtype == k.dtype == v.dtype
    scale = k.shape[-1] ** -0.5 if scale is None else scale
    o, final_state = ChunkDPLRDeltaRuleFunction.apply(
        q, # [B, T, H, K]
        k, # [B, T, H, K]
        v, # [B, T, H, K]
        a, # [B, T * RANK_AB, H, K]
        b, # [B, T * RANK_AB, H, K]
        gk, # [B, T, H, K]
        scale, 
        initial_state, # [B, T, H, K]
        output_final_state,

        RANK_AB, 
        head_first, 
        chunk_size, 
    )
    return o, final_state