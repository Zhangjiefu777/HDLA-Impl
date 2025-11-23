from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from fla.ops.utils.op import exp
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard, use_cuda_graph

@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BV in [16, 32, 64]
        for num_warps in [2, 4, 8, 16]
        for num_stages in [2, 3, 4]
    ],
    key=['BK'],
    use_cuda_graph=use_cuda_graph,
)
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_dplr_delta_rule_fwd_kernel(
    q,
    k,
    v,
    a,
    b,
    gk,
    o,
    h0,
    ht,
    cu_seqlens,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    RANK_AB: tl.constexpr, 
    REVERSE: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H

    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    
    time_offset = (bos + ((T - 1) if REVERSE else 0))
    p_q = q + time_offset * H*K + i_h * K + o_k
    p_k = k + time_offset * H*K + i_h * K + o_k
    p_a = a + time_offset * RANK_AB * H*K + i_h * K + o_k  # consider RANK_AB
    p_b = b + time_offset * RANK_AB * H*K + i_h * K + o_k  # consider RANK_AB
    p_gk = gk + time_offset * H*K + i_h * K + o_k
    p_v = v + time_offset * H*V + i_h * V + o_v
    p_o = o + time_offset * H*V + i_h * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * K*V + o_k[:, None] * V + o_v
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for _ in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32) * scale
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_gk = tl.load(p_gk, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)

        # Diag(decay) @ S_{t - 1}
        
        # TODO: ERROR!
        contribution_r = tl.zeros([BK, BV], dtype=tl.float32)
        # process the rank-r term of decay matrix
        # notice that [a1, a2] @ [b1, b2].T = a1 @ b1.T + a2 @ b2.T
        for r in range(RANK_AB):
            # an iteration steps computes: 
            # a_i @ b_i.T @ S_{t - 1}
            b_a_r = tl.load(p_a + r * H * K, mask=mask_k, other=0).to(tl.float32)
            b_b_r = tl.load(p_b + r * H * K, mask=mask_k, other=0).to(tl.float32)
            contribution_r += b_b_r[:, None] * tl.sum(b_a_r[:, None] * b_h, 0)[None, :]
        
        b_h = exp(b_gk)[:, None] * b_h
        b_h += contribution_r
        b_h += b_k[:, None] * b_v[None, :]
        
        b_o = tl.sum(b_h * b_q[:, None], 0)

        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
        
        step = (-1 if REVERSE else 1)
        p_q += step * H*K
        p_k += step * H*K
        p_a += step * RANK_AB * H * K # consider RANK_AB
        p_b += step * RANK_AB * H * K # consider RANK_AB
        p_gk += step * H*K
        p_v += step * H*V
        p_o += step * H*V

    if STORE_FINAL_STATE:
        p_ht = ht + i_nh * K*V + o_k[:, None] * V + o_v
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)

def fused_recurrent_dplr_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    gk: torch.Tensor,
    rank_ab: int,
    scale: Optional[float] = 1.0,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    reverse: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    # Extract shapes - a and b have merged T*RANK_AB dimension
    B, TR, H, K = a.shape
    T = TR // rank_ab  # Calculate original T from merged dimension
    
    # Validate merged dimension
    assert TR == T * rank_ab, f"Merged dimension TR={TR} must be divisible by rank_ab={rank_ab}"
    assert b.shape == (B, TR, H, K), f"b shape {b.shape} doesn't match a shape {a.shape}"
    assert k.shape == (B, T, H, K), f"k shape {k.shape} should be (B, T, H, K)"
    assert v.shape == (B, T, H, v.shape[-1]), f"v shape {v.shape} should be (B, T, H, V)"
    assert q.shape == (B, T, H, K), f"q shape {q.shape} should be (B, T, H, K)"
    assert gk.shape == (B, T, H, K), f"gk shape {gk.shape} should be (B, T, H, K)"

    V = v.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK = triton.next_power_of_2(K)

    h0 = initial_state
    ht = q.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None
    o = torch.empty_like(v)

    def grid(meta): return (triton.cdiv(V, meta['BV']), N * H)
    fused_recurrent_dplr_delta_rule_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        gk=gk,
        o=o,
        h0=h0,
        ht=ht,
        cu_seqlens=cu_seqlens,
        scale=scale,
        T=T,
        B=B,
        H=H,
        K=K,
        V=V,
        BK=BK,
        RANK_AB=rank_ab,  # Pass rank_ab as compile-time constant
        REVERSE=reverse,
    )
    return o, ht


class FusedRecurrentDPLRDeltaRuleFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        gk: torch.Tensor,
        rank_ab: int,
        scale: Optional[float] = None,
        initial_state: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
        reverse: bool = False,
        cu_seqlens: Optional[torch.LongTensor] = None,
    ):
        o, ht = fused_recurrent_dplr_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            gk=gk,
            rank_ab=rank_ab,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            reverse=reverse,
            cu_seqlens=cu_seqlens,
        )
        return o, ht

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dht):
        raise NotImplementedError(
            "Backward pass for fused_recurrent_dplr_delta_rule is not implemented and will not be supported. "
            "This kernel is only for inference. "
            "For training, please use `chunk_dplr_delta_rule`."
        )


def fused_recurrent_dplr_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    gk: torch.Tensor,
    rank_ab: int = 1,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    reverse: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    head_first: bool = False, 
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    This function computes the recurrence S_t = (I - Σ_{r=1}^{RANK_AB} a_t^{(r)} b_t^{(r)⊤})S_{t-1} + k_t v_t^⊤ 
    in a recurrent manner with rank-RANK_AB updates.

    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]`.
        a (torch.Tensor):
            a of shape `[B, T*RANK_AB, H, K]` with merged time and rank dimensions.
        b (torch.Tensor):
            b of shape `[B, T*RANK_AB, H, K]` with merged time and rank dimensions.
        gk (torch.Tensor):
            gk of shape `[B, T, H, K]`. decay term in log space!
        rank_ab (int):
            Rank of the update matrices (number of rank-1 components).
        scale (Optional[float]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        reverse (Optional[bool]):
            If `True`, process the state passing in reverse order. Default: `False`.
        cu_seqlens (Optional[torch.Tensor]):
            Cumulative sequence lengths of shape `[N + 1]` used for variable-length training,
            consistent with the FlashAttention API.
    """
    assert head_first == False, ""

    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )
    
    # Validate a and b shapes for merged T*RANK_AB dimension
    if len(a.shape) != 4:
        raise ValueError(f"a must have 4 dimensions [B, T*RANK_AB, H, K], got {len(a.shape)}")
    if len(b.shape) != 4:
        raise ValueError(f"b must have 4 dimensions [B, T*RANK_AB, H, K], got {len(b.shape)}")
    
    # Validate merged dimension
    B, TR, _, _ = a.shape
    T_expected = TR // rank_ab
    assert TR == T_expected * rank_ab, f"Merged dimension TR={TR} must be divisible by rank_ab={rank_ab}"
    assert q.shape[1] == T_expected, f"q time dimension {q.shape[1]} must match T={T_expected} from merged dimension"
    
    if scale is None:
        scale = q.shape[-1] ** -0.5
    
    o, final_state = FusedRecurrentDPLRDeltaRuleFunction.apply(
        q,
        k,
        v,
        a,
        b,
        gk,
        rank_ab,
        scale,
        initial_state,
        output_final_state,
        reverse,
        cu_seqlens,
    )
    return o, final_state