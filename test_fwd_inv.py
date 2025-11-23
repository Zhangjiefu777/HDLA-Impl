import triton
import triton.language as tl

import torch
import torch.nn.functional as F

from HRDPLR.fwd_triton import (
    fwd_prepare_wy_repr_kernel_chunk32,
)

@triton.jit(do_not_specialize=['T'])
def prepare_wy_repr_fwd_kernel_chunk32_ref(
    A_ab,
    A_ab_inv,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr, 
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    p_Aab = tl.make_block_ptr(A_ab + (bos*H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    p_Aab_inv = tl.make_block_ptr(A_ab_inv + (bos*H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_A_ab = tl.load(p_Aab, boundary_check=(0, 1))
    b_A_ab = tl.where(tl.arange(0, BT)[:, None] > tl.arange(0, BT)[None, :], b_A_ab, 0)
    for i in range(1, BT):
        mask = tl.arange(0, BT) == i
        b_a = tl.sum(tl.where(mask[:, None], b_A_ab, 0), 0)
        b_a = b_a + tl.sum(b_a[:, None] * b_A_ab, 0) * (tl.arange(0, BT) < i)
        b_A_ab = tl.where(mask[:, None], b_a, b_A_ab)
    b_A_ab += tl.arange(0, BT)[:, None] == tl.arange(0, BT)[None, :]
    tl.store(p_Aab_inv, b_A_ab.to(p_Aab_inv.dtype.element_ty), boundary_check=(0, 1))

if __name__ == '__main__':
    B = 2
    H = 4
    T = 256
    K = 128
    V = 64
    RANK_AB = 2

    BT = 16
    BK = K
    BV = 16
    BC = BT
    
    NT = triton.cdiv(T, BT)
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)

    device = torch.device('cuda')
    dtype = torch.float32

    A_ab = torch.randn(
        size=(B, T * RANK_AB, H, BT * RANK_AB), device=device, dtype=dtype
    )
    A_ab_inv_custom = torch.zeros_like(A_ab)
    A_ab_inv_ref = torch.zeros_like(A_ab)

    grid = [NT, B * H]
    fwd_prepare_wy_repr_kernel_chunk32[grid](
        A_ab, # [B, H, T * RANK_AB, BT * RANK_AB]
        A_ab_inv_custom, # [B, H, T * RANK_AB, BT * RANK_AB]
        B, 
        H, 
        T, 
        BT, 
        RANK_AB, 
    )

    grid = [NT, B * H]
    prepare_wy_repr_fwd_kernel_chunk32_ref[grid](
        A_ab, 
        A_ab_inv_ref, 
        B, 
        H, 
        T * RANK_AB, 
        H, 
        BT * RANK_AB, 
        BT, 
        False, 
    )

    assert torch.allclose(A_ab_inv_ref, A_ab_inv_custom, rtol=5e-2, atol=5e-2)