import torch
import torch.nn.functional as F

import triton

from typing import Tuple

from einops import rearrange


def fwd_h_naive(
    kg: torch.Tensor, 
    v: torch.Tensor, 
    bg: torch.Tensor,
    w: torch.Tensor, 
    u: torch.Tensor,  
    gk: torch.Tensor, # cumsum(gi)

    h0: torch.Tensor,
    BT: int, 
    RANK_AB: int, 
):
    B, T, H, K, V = *kg.shape, v.shape[-1]
    NT = triton.cdiv(T, BT)

    device = kg.device
    dtype = kg.dtype

    h_ref = torch.zeros(
        size=(B, NT, H, K, V), device=device, dtype=dtype, 
    )
    v_new_ref = torch.zeros_like(u)
    
    h_ref[:, 0, ...] = h0
    for i_t in range(NT):
        gk_last = gk[:, (i_t + 1) * BT - 1, ...] # [B, H, K]
        v_new_ref[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...] = (
            torch.einsum(
                'b c h k, b h k v -> b c h v', # A @ B 
                w[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...], 
                h_ref[:, i_t, ...]
            ) + u[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...] 
        )
        # [B, H, K, V]
        if i_t < NT - 1:
            h_ref[:, i_t + 1, ...] = (
                (
                    torch.exp(gk_last)[..., None] * h_ref[:, i_t, ...]
                ) + torch.einsum(
                    'b c h k, b c h v -> b h k v', # A.T @ B 
                    kg[:, i_t * BT: (i_t + 1) * BT, ...], # [B, BT, H, K]
                    v[:, i_t * BT: (i_t + 1) * BT, ...], # [B, BT, H, V]
                ) + torch.einsum(
                    'b c h k, b c h v -> b h k v', # A.T @ B 
                    bg[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...], # [B, BT * RANK_AB, H, K]
                    v_new_ref[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...], # [B, BT * RANK_AB, H, V]
                )
            )
        else:
            ht_ref = (
                (
                    torch.exp(gk_last)[..., None] * h_ref[:, i_t, ...]
                ) + torch.einsum(
                    'b c h k, b c h v -> b h k v', # A.T @ B 
                    kg[:, i_t * BT: (i_t + 1) * BT, ...], # [B, BT, H, K]
                    v[:, i_t * BT: (i_t + 1) * BT, ...], # [B, BT, H, V]
                ) + torch.einsum(
                    'b c h k, b c h v -> b h k v', # A.T @ B 
                    bg[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...], # [B, BT * RANK_AB, H, K]
                    v_new_ref[:, i_t * BT * RANK_AB: (i_t + 1) * BT * RANK_AB, ...], # [B, BT * RANK_AB, H, V]
                )
            )
    return h_ref, ht_ref, v_new_ref


def hrdplr_naive(
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
    dtype = q.dtype
    device = q.device

    B, T, H, K, V = *q.shape, v.shape[-1]
    BT = chunk_size
    NT = triton.cdiv(T, BT)

    q, k, v, gk = map(
        lambda x: rearrange(x, 'b (n c) h d -> b h n c d', c=BT), 
        [q, k, v, gk]
    ) # [B, H, NT, BT, D]
    a, b = map(
        lambda x: rearrange(x, 'b (n c) h d -> b h n c d', c=BT * RANK_AB), 
        [a, b]
    )

    gi = gk.cumsum(-2)
    ge = gi - gk

    gi_ab = torch.repeat_interleave(gi, repeats=RANK_AB, dim=-2)
    ge_ab = torch.repeat_interleave(ge, repeats=RANK_AB, dim=-2)

    A_qk = torch.zeros(
        size=(B, H, NT, BT, BT), device=device, dtype=dtype, 
    )
    A_qb = torch.zeros(
        size=(B, H, NT, BT, BT * RANK_AB), device=device, dtype=dtype, 
    )
    A_ak = torch.zeros(
        size=(B, H, NT, BT * RANK_AB, BT), device=device, dtype=dtype, 
    )
    A_ab = torch.zeros(
        size=(B, H, NT, BT * RANK_AB, BT * RANK_AB), device=device, dtype=dtype, 
    )
    o = torch.zeros_like(v)
    S = torch.zeros(
        size=(B, H, K, V), device=device, dtype=dtype, 
    )

    range_BT = torch.arange(0, BT, device=device)
    range_BT_AB = torch.repeat_interleave(
        input=range_BT, repeats=RANK_AB, dim=0, 
    )

    # Attention scores 
    for i in range(chunk_size):
        q_i = q[:, :, :, i: i + 1, ...] # [B, H, NT, 1, K]
        gi_i = gi[:, :, :, i: i + 1, ...] # [B, H, NT, 1, K]
        ge_i = ge[:, :, :, i: i + 1, ...] # [B, H, NT, 1, K]

        mask_qk = (range_BT <= i) # tril(..., 0), [BT, ]
        mask_qb = (range_BT_AB <= i) # tril(..., 0), [BT * RANK_AB, ]

        mask_ak = (range_BT < i) # tril(..., -1), [BT, ]
        mask_ab = (range_BT_AB < i) # tril(..., -1), [BT * RANK_AB, ]

        # TODO: causal mask () + decay term

        A_qk[:, :, :, i, :] = torch.sum(
            (
                q_i * k # [B, H, NT, 1, K] * [B, H, NT, BT, K] -> [B, H, NT, BT, K] 
            ) * torch.exp(
                gi_i - gi # [B, H, NT, 1, K] - [B, H, NT, BT, K] -> [B, H, NT, BT, K]
            ) * mask_qk[None, None, None, :, None], # [1, 1, 1, BT, 1] 
            dim=-1, # [B, H, NT, BT, K] -> [B, H, NT, BT] 
        )

        A_qb[:, :, :, i, :] = torch.sum(
            (
                q_i * b # [B, H, NT, 1, K] * [B, H, NT, BT * RANK_AB, K] -> [B, H, NT, BT * RANK_AB, K] 
            ) * torch.exp(
                gi_i - gi_ab # [B, H, NT, 1, K] - [B, H, NT, BT * RANK_AB, K] -> [B, H, NT, BT * RANK_AB, K]
            ) * mask_qb[None, None, None, :, None], # [1, 1, 1, BT * RANK_AB, 1] 
            dim=-1, # [B, H, NT, BT * RANK_AB, K] -> [B, H, NT, BT * RANK_AB] 
        ) # [B, H, NT, BT * RANK_AB]

        for i_r in range(RANK_AB):
            a_ir = a[:, :, :, i * RANK_AB + i_r: i * RANK_AB + i_r + 1, ...]
            A_ak[:, :, :, i * RANK_AB + i_r, :] = torch.sum(
                (
                    a_ir * k # [B, H, NT, 1, K] * [B, H, NT, BT, K] -> [B, H, NT, BT, K] 
                ) * torch.exp(
                    ge_i - gi # [B, H, NT, 1, K] - [B, H, NT, BT, K] -> [B, H, NT, BT, K]
                ) * mask_ak[None, None, None, :, None], # [1, 1, 1, BT, 1] 
                dim=-1, # [B, H, NT, BT, K] -> [B, H, NT, BT] 
            )
            A_ab[:, :, :, i * RANK_AB + i_r, :] = torch.sum(
                (
                    a_ir * b # [B, H, NT, 1, K] * [B, H, NT, BT * RANK_AB, K] -> [B, H, NT, BT * RANK_AB, K] 
                ) * torch.exp(
                    ge_i - gi_ab # [B, H, NT, 1, K] - [B, H, NT, BT * RANK_AB, K] -> [B, H, NT, BT * RANK_AB, K]
                ) * mask_ab[None, None, None, :, None], # [1, 1, 1, BT * RANK_AB, 1] 
                dim=-1, # [B, H, NT, BT * RANK_AB, K] -> [B, H, NT, BT * RANK_AB] 
            ) # [B, H, NT, BT * RANK_AB]

    A_ab_original = A_ab.clone()

    # The inverse of A_ab
    for i in range(RANK_AB, BT * RANK_AB):
        A_ab[..., i, :i] = A_ab[..., i, :i].clone() + (A_ab[..., i, :, None].clone() * A_ab[..., :, :i].clone()).sum(-2)

    A_ab += torch.eye(BT * RANK_AB, dtype=torch.float32, device=device)[None, None, None, ...]

    # WY Representation
    w = A_ab @ (a * torch.exp(ge_ab)) # [BT * RANK_AB, K]
    u = A_ab @ (A_ak @ v) # [BT * RANK_AB, BT * RANK_AB] @ ([BT * RANK_AB, BT] @ [BT, V]) -> [BT * RANK_AB, V]

    S = initial_state

    bg = torch.empty_like(b)
    kg = torch.empty_like(k)

    for i_t in range(NT):
        q_i, k_i, v_i, w_i, u_i, b_i = q[:, :, i_t], k[:, :, i_t], v[:, :, i_t], w[:, :, i_t], u[:, :, i_t], b[:, :, i_t]
        gi_i = gi[:, :, i_t]

        lambda_tilde = torch.exp(
            gi_i[:, :, -1: , :] - gi_i
        ) # [B, H, 1, K] - [B, H, BT, K]
        lambda_tilde_ab = torch.repeat_interleave(
            lambda_tilde, repeats=RANK_AB, dim=-2, 
        )

        kg[:, :, i_t] = k_i * lambda_tilde
        bg[:, :, i_t] = b_i * lambda_tilde_ab

    # version 2 of hidden state checkpoint computation, for debug
    # kg, v, gi = map(
    #     lambda x: rearrange(x, 'b h n r c -> b (n r) h c'), 
    #     [kg, v, gi]
    # )
    # bg, w, u = map(
    #     lambda x: rearrange(x, 'b h n r c -> b (n r) h c'), 
    #     [bg, w, u]
    # )

    # h_ref, ht_ref, v_new_ref = fwd_h_naive(
    #     kg, v, bg, w, u, gi, initial_state, BT, RANK_AB, 
    # )

    # Hidden State Checkpoints & Linear Attention Output
    for i_t in range(NT):
        q_i, k_i, v_i, w_i, u_i, b_i = q[:, :, i_t], k[:, :, i_t], v[:, :, i_t], w[:, :, i_t], u[:, :, i_t], b[:, :, i_t]
        gi_i = gi[:, :, i_t]

        v2_i = w_i @ S + u_i # [BT * RANK_AB, K] @ [K, V] + [BT * RANK_AB, V]

        o_qkv = A_qk[:, :, i_t] @ v_i
        o_qbv2 = A_qb[:, :, i_t] @ v2_i
        o_inter = (q_i * torch.exp(gi_i)) @ S

        o[:, :, i_t] = o_qkv + o_qbv2 + o_inter

        lambda_tilde = torch.exp(
            gi_i[:, :, -1: , :] - gi_i
        ) # [B, H, 1, K] - [B, H, BT, K]
        lambda_tilde_ab = torch.repeat_interleave(
            lambda_tilde, repeats=RANK_AB, dim=-2, 
        )

        S = (
            S * torch.exp(gi_i[:, :, -1, :, None]) # [B, H, K, V] * [B, H, K, 1]
            + kg[:, :, i_t].transpose(-2, -1) @ v_i
            + bg[:, :, i_t].transpose(-2, -1) @ v2_i
        )
    S = None if output_final_state is False else S

    A_qk, A_qb, A_ak, A_ab_original, A_ab_inv = map(
        lambda x: rearrange(x, 'b h n r c -> b (n r) h c'), 
        [A_qk, A_qb, A_ak, A_ab_original, A_ab]
    )

    return rearrange(o, 'b h n c d -> b (n c) h d'), S #, w, u, A_qk, A_qb, A_ak, A_ab_original, A_ab_inv, kg, bg #, h_ref, ht_ref, v_new_ref