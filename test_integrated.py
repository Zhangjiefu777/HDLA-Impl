import triton
import torch
import torch.nn.functional as F

from HRDPLR.hrdplr import chunk_hrdplr_delta_rule
from HRDPLR.naive import hrdplr_naive

if __name__ == '__main__':
    B = 2
    H = 4
    T = 256
    K = 128
    V = 64
    RANK_AB = 2

    BT = 16

    device = torch.device('cuda')
    dtype = torch.float32

    q = torch.randn(size=(B, T, H, K), device=device, dtype=dtype, requires_grad=True)
    
    k = torch.randn(size=(B, T, H, K), device=device, dtype=dtype)
    k /= (torch.norm(k, dim=-1, keepdim=True) + 1e-6)
    k.requires_grad_()

    v = torch.randn(size=(B, T, H, V), device=device, dtype=dtype, requires_grad=True)
    
    a = torch.randn(size=(B, T * RANK_AB, H, K), device=device, dtype=dtype)
    a /= (torch.norm(a, dim=-1, keepdim=True) + 1e-6)
    a.requires_grad_()

    b = torch.randn(size=(B, T * RANK_AB, H, K), device=device, dtype=dtype)
    b /= (torch.norm(b, dim=-1, keepdim=True) + 1e-6)
    b.requires_grad_()

    gk = F.logsigmoid(
        torch.randn(size=(B, T, H, K), device=device, dtype=torch.float32),
    )
    gk.requires_grad_()

    initial_state = torch.randn(size=(B, H, K, V), device=device, dtype=dtype, requires_grad=True)

    o, final_state = chunk_hrdplr_delta_rule(
        q, k, v, a, b, gk, RANK_AB, 
        scale=1.0, # equivalent to apply row-wise on q
        initial_state=initial_state, # [B, H, K, V]
        output_final_state=True, 
        head_first=False, 
        chunk_size=BT, 
    )

    do = torch.randn_like(o)
    dht = torch.randn_like(final_state)

    dq, dk, dv, da, db, dgk, dh0 = torch.autograd.grad(
        outputs=[o, final_state], 
        inputs=[q, k, v, a, b, gk, initial_state], 
        grad_outputs=[do, dht], 
        retain_graph=True, 
        create_graph=True, 
    )

    q.grad = None
    k.grad = None
    v.grad = None
    a.grad = None
    b.grad = None
    gk.grad = None
    initial_state.grad = None

    o_ref, final_state_ref = hrdplr_naive(
        q, k, v, a, b, gk, 1.0, initial_state, True, RANK_AB, False, chunk_size=BT, 
    )

    dq_ref, dk_ref, dv_ref, da_ref, db_ref, dgk_ref, dh0_ref = torch.autograd.grad(
        outputs=[o_ref, final_state_ref],
        inputs=[q, k, v, a, b, gk, initial_state],
        grad_outputs=[do, dht],
        retain_graph=True,
        create_graph=True
    )

    assert torch.allclose(o, o_ref, rtol=5e-2, atol=5e-2)
    assert torch.allclose(final_state, final_state_ref, rtol=5e-2, atol=5e-2)

    assert torch.allclose(dq, dq_ref, rtol=5e-2, atol=5e-2)
    assert torch.allclose(dk, dk_ref, rtol=5e-2, atol=5e-2)
    assert torch.allclose(dv, dv_ref, rtol=5e-2, atol=5e-2)
    assert torch.allclose(da, da_ref, rtol=5e-2, atol=5e-2)
    assert torch.allclose(db, db_ref, rtol=5e-2, atol=5e-2)
    assert torch.allclose(dgk, dgk_ref, rtol=5e-2, atol=5e-2)
    assert torch.allclose(dh0, dh0_ref, rtol=5e-2, atol=5e-2)

    print("Test passed!")