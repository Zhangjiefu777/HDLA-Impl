from HRDPLR.hrdplr import chunk_hrdplr_fwd
from fla.ops.generalized_delta_rule.dplr.chunk import chunk_dplr_fwd

import torch
import torch.nn.functional as F

import triton

if __name__ == '__main__':
    seed = 77
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    B = 2
    H = 2
    T = 1024
    K = 64
    V = 96

    RANK_AB = 2

    BT = 32
    NT = triton.cdiv(T, BT)

    device = torch.device('cuda:0')
    dtype = torch.bfloat16

    q = torch.randn(
        size=(B, T, H, K), device=device, dtype=dtype, 
    )
    k = torch.randn(
        size=(B, T, H, K), device=device, dtype=dtype, 
    )
    v = torch.randn(
        size=(B, T, H, V), device=device, dtype=dtype, 
    )
    a_1 = torch.randn(
        size=(B, T, H, K), device=device, dtype=dtype, 
    )
    a_2 = torch.randn(
        size=(B, T, H, K), device=device, dtype=dtype, 
    )
    b_1 = torch.randn(
        size=(B, T, H, K), device=device, dtype=dtype, 
    )
    b_2 = torch.randn(
        size=(B, T, H, K), device=device, dtype=dtype, 
    )

    k /= k.abs().max()
    v /= v.abs().max()
    a_1 /= a_1.abs().max()
    a_2 /= a_2.abs().max()
    b_1 /= b_1.abs().max()
    b_2 /= b_2.abs().max()

    g_sqrt = F.logsigmoid(
        torch.randn(
            size=(B, T, H, K), device=device, dtype=torch.float32, 
        )
    )
    gk_sqrt = torch.zeros_like(g_sqrt)
    for i_t in range(NT):
        gk_sqrt[:, i_t * BT: (i_t + 1) * BT, :, :] = torch.cumsum(
            g_sqrt[:, i_t * BT: (i_t + 1) * BT, :, :], dim=1, 
        ) 
    gk = gk_sqrt * 2


    zero_q = torch.zeros_like(q)
    zero_k = torch.zeros_like(k)
    zero_v = torch.zeros_like(v)

    q_stacked = torch.stack((zero_q, q), dim=2)
    q_padded = torch.reshape(q_stacked, (B, T * RANK_AB, H, K)).contiguous()

    k_stacked = torch.stack((zero_k, k), dim=2)
    k_padded = torch.reshape(k_stacked, (B, T * RANK_AB, H, K)).contiguous()

    v_stacked = torch.stack((zero_v, v), dim=2)
    v_padded = torch.reshape(v_stacked, (B, T * RANK_AB, H, V)).contiguous()

    gk_sqrt_repeated = torch.repeat_interleave(
        gk_sqrt, repeats=RANK_AB, dim=1
    ).contiguous()

    a_stacked = torch.stack((a_1, a_2), dim=2)
    a_rank1_iter = torch.reshape(a_stacked, (B, T * RANK_AB, H, K)).contiguous()

    b_stacked = torch.stack((b_1, b_2), dim=2)
    b_rank1_iter = torch.reshape(b_stacked, (B, T * RANK_AB, H, K)).contiguous()

    # The inner product of (b_{t, 2}, a_{t, 1})
    coeff = torch.einsum(
        'b t h k, b t h k -> b t h', 
        b_2, # [B, T, H, K]
        a_1, # [B, T, H, K]
    ) # [B, T, H]

    g_sqrt_exp = torch.exp(g_sqrt) # [B, T, H, K]
    a_1_gated = g_sqrt_exp * a_1
    b_2_gated = g_sqrt_exp * b_2

    a_1_merged = a_1_gated - coeff[..., None] * a_2
    a_2_merged = a_2

    a_merged_stacked = torch.stack((a_1_merged, a_2_merged), dim=2)
    a_merged = torch.reshape(a_merged_stacked, (B, T * RANK_AB, H, K)).contiguous()

    b_1_merged = b_1
    b_2_merged = b_2_gated

    b_merged_stacked = torch.stack((b_1_merged, b_2_merged), dim=2)
    b_merged = torch.reshape(b_merged_stacked, (B, T * RANK_AB, H, K)).contiguous()

    initial_state = torch.randn((B, H, K, V), device=device, dtype=dtype)

    o_ref, h_ref = chunk_dplr_fwd(
        q=q_padded,
        k=k_padded,
        v=v_padded,
        a=a_rank1_iter,
        b=b_rank1_iter,
        gk=gk_sqrt_repeated,
        scale=1,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=None,
        chunk_size=BT * RANK_AB, 
    )

    q_t = q.transpose(1, 2).contiguous()
    k_t = k.transpose(1, 2).contiguous()
    v_t = v.transpose(1, 2).contiguous()
    a_t = a_merged.transpose(1, 2).contiguous()
    b_t = b_merged.transpose(1, 2).contiguous()
    gk_t = gk.transpose(1, 2).contiguous()

    o_t, h = chunk_hrdplr_fwd(
        q=q_t, k=k_t, v=v_t, a=a_t, b=b_t, gk=gk_t, 
        scale=1, initial_state=initial_state, output_final_state=True, 
        RANK_AB=2, head_first=True, chunk_size=BT
    )
    o_ref = o_ref[:, 1::2, ...]
    o = o_t.transpose(1, 2)

    assert torch.allclose(o, o_ref)
    assert torch.allclose(h, h_ref)