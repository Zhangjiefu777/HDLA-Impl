import triton
import torch
import torch.nn.functional as F

from HRDPLR.fwd_triton import chunk_dplr_fwd_kernel_o

import torch

def chunk_dplr_fwd_o_simple(
    qg, # [B, T, H, K]
    v, # [B, T, H, V]
    v_new, # [B, T * RANK_AB, H, V]
    A_qk, # [B, T, H, BT]
    A_qb, # [B, T, H, BT * RANK_AB]
    h, # [B, NT, H, K, V]
    BT: int,
    RANK_AB: int,
):
    """
    
    Args:
        qg: [B, T, H, K] - Query gate tensor
        v: [B, T, H, V] - Value tensor
        v_new: [B, T * RANK_AB, H, V] - New value tensor
        A_qk: [B, T, H, BT] - Attention QK matrix
        A_qb: [B, T, H, BT * RANK_AB] - Attention QB matrix
        h: [B, NT, H, K, V] - Hidden state tensor
        BT: Block size for T dimension
        RANK_AB: Rank of AB matrix
    
    Returns:
        o: [B, T, H, V] - Output tensor
    """
    B, T, H, K = qg.shape
    _, _, _, V = v.shape
    BT_AB = BT * RANK_AB
    NT = triton.cdiv(T, BT)
    
    # Initialize output
    o = torch.zeros(B, T, H, V, device=qg.device, dtype=qg.dtype)
    
    # Process each chunk in T dimension
    for i_t in range(NT):
        # Calculate current chunk boundaries
        t_start = i_t * BT
        t_end = min(t_start + BT, T)
        current_BT = t_end - t_start
        
        # Get chunk slices
        qg_chunk = qg[:, t_start:t_end, :, :]  # [B, current_BT, H, K]
        h_chunk = h[:, i_t, :, :, :] # [B, H, K, V]
        A_qk_chunk = A_qk[:, t_start:t_end, :, :]  # [B, current_BT, H, BT]
        v_chunk = v[:, t_start:t_end, :, :]  # [B, current_BT, H, V]
        A_qb_chunk = A_qb[:, t_start:t_end, :, :]  # [B, current_BT, H, BT_AB]
        
        # Get v_new chunk
        v_new_start = i_t * BT_AB
        v_new_end = v_new_start + BT_AB
        v_new_chunk = v_new[:, v_new_start:v_new_end, :, :]  # [B, BT_AB, H, V]
        
        # Batch matrix multiplication
        term1 = torch.einsum('bthk,bhkv->bthv', qg_chunk, h_chunk)  # [B, current_BT, H, V]
        
        # Step 2: Calculate A_qk @ v - [B, current_BT, H, V]
        # Create lower triangular mask
        mask_qk = torch.tril(torch.ones(current_BT, current_BT, device=qg.device), diagonal=0)  # [current_BT, current_BT]
        mask_qk = mask_qk.unsqueeze(0).unsqueeze(2)  # [1, current_BT, 1, current_BT]
        
        # Apply mask to A_qk
        A_qk_masked = A_qk_chunk * mask_qk  # [B, current_BT, H, current_BT]
        
        # Batch matrix multiplication
        term2 = torch.einsum('brhm,bmhc->brhc', A_qk_masked, v_chunk)  # [B, current_BT, H, V]
        
        # Step 3: Calculate A_qb @ v_new - [B, current_BT, H, V]
        # Create lower triangular mask for AB
        o_s_kv = torch.arange(current_BT, device=qg.device)
        o_s_ab = torch.repeat_interleave(o_s_kv, RANK_AB)  # [BT_AB]
        mask_qb = (o_s_kv[:, None] >= o_s_ab[None, :]).float()  # [current_BT, BT_AB]
        mask_qb = mask_qb.unsqueeze(0).unsqueeze(2)  # [1, current_BT, 1, BT_AB]
        
        # Apply mask to A_qb
        A_qb_masked = A_qb_chunk * mask_qb  # [B, current_BT, H, BT_AB]
        
        # Batch matrix multiplication
        term3 = torch.einsum('brhm,bmhc->brhc', A_qb_masked, v_new_chunk)  # [B, current_BT, H, V]
        
        # Combine all terms
        output_chunk = term1 + term2 + term3
        
        # Store result to output
        o[:, t_start:t_end, :, :] = output_chunk
    
    return o

if __name__ == '__main__':

    B = 2
    H = 4
    T = 256
    K = 96
    V = 64
    RANK_AB = 2

    BT = 16
    
    NT = triton.cdiv(T, BT)


    device = torch.device('cuda')
    dtype = torch.float32

    qg = torch.ones(size=(B, T, H, K), device=device, dtype=dtype)
    v = torch.randn(size=(B, T, H, V), device=device, dtype=dtype)
    v_new = torch.randn(size=(B, T * RANK_AB, H, V), device=device, dtype=dtype)
    A_qk = torch.zeros(size=(B, T, H, BT), device=device, dtype=dtype)
    A_qb = torch.zeros(size=(B, T, H, BT * RANK_AB), device=device, dtype=dtype)
    h = torch.randn(size=(B, NT, H, K, V), device=device, dtype=dtype)

    o_triton = torch.zeros(size=(B, T, H, V), device=device, dtype=dtype)
    o_torch = torch.zeros(size=(B, T, H, V), device=device, dtype=dtype)

    def grid(meta): return [triton.cdiv(V, meta['BV']), NT, B * H]
    chunk_dplr_fwd_kernel_o[grid](
        qg, # old: [B, H, T, K], new: [B, T, H, K]
        v, # old: [B, H, T, V], new: [B, T, H, V]
        v_new, # old: [B, H, T * RANK_AB, V], new: [B, T * RANK_AB, H, V]
        A_qk, # old: [B, H, T, BT], new: [B, T, H, BT]
        A_qb, # old: [B, H, T, BT * RANK_AB], new: [B, T, H, BT * RANK_AB]
        h, # old:  [B, H, NT, K, V], new: [B, NT, H, K, V]
        o_triton, # old: [B, H, T, V], new: [B, T, H, V]
        T,
        H,
        K,
        V,
        BT,
        RANK_AB, 
        BT * RANK_AB, # BT_AB = BT * RANK_AB
    )

    o_torch = chunk_dplr_fwd_o_simple(
        qg, # [B, T, H, K]
        v, # [B, T, H, V]
        v_new, # [B, T * RANK_AB, H, V]
        A_qk, # [B, T, H, BT]
        A_qb, # [B, T, H, BT * RANK_AB]
        h, # [B, NT, H, K, V]
        BT,
        RANK_AB,
    )

    assert torch.allclose(o_triton, o_torch, rtol=5e-2, atol=5e-2)