import torch
import torch.nn.functional as F

from einops import rearrange

from HRDPLR import (
    chunk_hrdplr_delta_rule, 
    fused_recurrent_dplr_delta_rule, 
    hrdplr_naive, 
)

from fla.ops import (
    chunk_dplr_delta_rule, 
)

def hdla_custom(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor, 
    log_f: torch.Tensor, 
    beta: torch.Tensor, 
    initial_state: torch.Tensor, 
    causal: bool = True, 
    training: bool = True, 
    use_cache: bool = True, 
):
    a_t1 = -beta * k # [B, T, H, 1] * [B, T, H, K]
    a_t2 = -beta * torch.exp(log_f) * k # [B, T, H, 1] * [B, T, H, K] * [B, T, H, K]

    b_t1 = k
    b_t2 = k

    a_1 = a_t1 # [B, T, H, K]
    a_2 = a_t2 + a_t1 * torch.sum(
        b_t1 * a_t2, dim=-1, keepdim=True, # [B, T, H, K] -> [B, T, H, 1] 
    ) # [B, T, H, K]
    
    b_1 = torch.exp(log_f) * b_t1
    b_2 = b_t2

    a_stacked = torch.stack([a_1, a_2], dim=2) # [B, T, 2, H, K]
    b_stacked = torch.stack([b_1, b_2], dim=2) # [B, T, 2, H, K]

    a = rearrange(a_stacked, 'b t r h k -> b (t r) h k')
    b = rearrange(b_stacked, 'b t r h k -> b (t r) h k')

    scale = 1.
    if causal:
        dtype = q.dtype
        if training or use_cache:
            output, recurrent_state = chunk_hrdplr_delta_rule(
                q=q,
                k=k.to(dtype),
                v=v.to(dtype),
                a=a.to(dtype),
                b=b.to(dtype),
                gk=log_f.to(dtype),
                RANK_AB=2, 
                initial_state=initial_state,
                output_final_state=use_cache,
                scale=scale,
                head_first=False,
                chunk_size=16, 
            )
        else:
            output, recurrent_state = fused_recurrent_dplr_delta_rule(
                q=q,
                k=k.to(dtype),
                v=v.to(dtype),
                a=a.to(dtype),
                b=b.to(dtype),
                gk=log_f.to(dtype),
                rank_ab=2, 
                initial_state=initial_state,
                output_final_state=use_cache,
                scale=scale,
                head_first=False,
                chunk_size=16, 
            )
    else:
        assert False

    return output, recurrent_state

def hdla_fla(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor, 
    log_f: torch.Tensor, 
    beta: torch.Tensor, 
    initial_state: torch.Tensor, 
    causal: bool = True, 
    training: bool = True, 
    use_cache: bool = True, 
):
    n = q.shape[1]
    zero_qk = torch.zeros_like(q)
    zero_v = torch.zeros_like(v)

    k_ = k * (-beta)
    a = torch.cat([k_ * torch.exp(log_f), k_], dim=1)
    b = torch.cat([k, k], dim=1)
    q = torch.cat([zero_qk, q], dim=1)
    k = torch.cat([zero_qk, k], dim=1)
    v = torch.cat([zero_v, v], dim=1)
    log_f = torch.cat([log_f, zero_qk], dim=1)
    log_f, a, b, k, v, q = map(
        lambda x: rearrange(x, "b (c n) ... -> b (n c) ... ", c=2),
        [log_f, a, b, k, v, q],
    )

    scale = 1.
    if causal:
        dtype = q.dtype
        if training or use_cache:
            fn = chunk_dplr_delta_rule
        else:
            fn = fused_recurrent_dplr_delta_rule

        output, recurrent_state = fn(
            q=q,
            k=k.to(dtype),
            v=v.to(dtype),
            a=a.to(dtype),
            b=b.to(dtype),
            gk=log_f.to(dtype),
            initial_state=initial_state,
            output_final_state=use_cache,
            scale=scale,
            head_first=False,
        )
    else:
        assert False

    output = rearrange(output, "b (n c) ... -> b (c n) ...", c=2)
    output = output[:, -n:]

    return output, recurrent_state

def hdla_naive(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor, 
    log_f: torch.Tensor, 
    beta: torch.Tensor, 
    initial_state: torch.Tensor, 
    causal: bool = True, 
    training: bool = True, 
    use_cache: bool = True, 
):
    a_t1 = -beta * k # [B, T, H, 1] * [B, T, H, K]
    a_t2 = -beta * torch.exp(log_f) * k # [B, T, H, 1] * [B, T, H, K] * [B, T, H, K]

    b_t1 = k
    b_t2 = k

    a_1 = a_t1 # [B, T, H, K]
    a_2 = a_t2 + a_t1 * torch.sum(
        b_t1 * a_t2, dim=-1, keepdim=True, # [B, T, H, K] -> [B, T, H, 1] 
    ) # [B, T, H, K]
    
    b_1 = torch.exp(log_f) * b_t1
    b_2 = b_t2

    a_stacked = torch.stack([a_1, a_2], dim=2) # [B, T, 2, H, K]
    b_stacked = torch.stack([b_1, b_2], dim=2) # [B, T, 2, H, K]

    a = rearrange(a_stacked, 'b t r h k -> b (t r) h k')
    b = rearrange(b_stacked, 'b t r h k -> b (t r) h k')

    return hrdplr_naive(
        q, k, v, a, b, log_f, 
        scale=1., initial_state=initial_state, output_final_state=True, 
        RANK_AB=2, head_first=False, chunk_size=16, 
    )

if __name__ == '__main__':
    B = 2
    T = 256
    H = 4
    K = 128
    V = 64

    dtype = torch.float32
    device = torch.device('cuda')

    q = torch.randn(
        size=(B, T, H, K), device=device, dtype=dtype, 
    )

    k = torch.randn(
        size=(B, T, H, K), device=device, dtype=dtype, 
    )
    k /= (torch.norm(k, dim=-1, keepdim=True) + 1e-5)

    v = torch.randn(
        size=(B, T, H, V), device=device, dtype=dtype, 
    )
    f = F.sigmoid(
        torch.randn(
            size=(B, T, H, K), device=device, dtype=torch.float32, 
        )
    )
    log_f = torch.log(f)

    # for debug:
    # log_f.zero_() # case 1: no diagonal decay
    # log_f -= 0.03 # case 2: (closed to 1) constant diagonal decay
    # log_f -= 1.00 # case 3: (closed to 0.4) constant diagonal decay 

    beta = 2 * F.sigmoid(
        torch.randn(
            size=(B, T, H, K), device=device, dtype=torch.float32, 
        )
    )
    initial_state = torch.randn(
        size=(B, H, K, V), device=device, dtype=dtype, 
    )

    o_fla, ht_fla = hdla_fla(
        q, k, v, log_f, beta, initial_state
    )
    o_custom, ht_custom = hdla_custom(
        q, k, v, log_f, beta, initial_state
    )

    o_naive, ht_naive = hdla_naive(
        q, k, v, log_f, beta, initial_state
    )

    assert torch.allclose(o_fla, o_custom, rtol=5e-2, atol=5e-2)
    assert torch.allclose(ht_fla, ht_custom, rtol=5e-2, atol=5e-2)
    assert torch.allclose(o_custom, o_naive, rtol=5e-2, atol=5e-2)