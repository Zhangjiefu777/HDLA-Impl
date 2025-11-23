"""
This file is used to test the speed of 2 implementations on
Diagonal-Plus-Rank-2 decay + Rank-1 key-value update. 

The naive implementation needs to extend the timesteps of q/k/v from T to 2T, 
and then call the diagonal-plus-rank-1 kernel in fla repository. 

The customized implementation uses our newly implemented Triton kernel.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn.functional as F

import triton

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import Optional
from einops import rearrange


from fla.ops.generalized_delta_rule.dplr import chunk_dplr_delta_rule
from HRDPLR.hrdplr import chunk_hrdplr_delta_rule

def run_dplr(
    q: torch.Tensor, 
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    gk: torch.Tensor,
    scale: Optional[int] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False,
):
    output, final_state = chunk_dplr_delta_rule(
        q,
        k,
        v,
        a,
        b,
        gk,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        head_first,
    )
    # (3) output selection
    # output = rearrange(output, "b (n c) ... -> b (c n) ...", c=2)
    # output = output[:, -n:]
    # output = rearrange(output, "b n h d -> b n (h d)")
    
    return output

def run_hrdplr(
    q: torch.Tensor, # [B, H, T, K]
    k: torch.Tensor, # [B, H, T, K]
    v: torch.Tensor, # [B, H, T, V]
    a: torch.Tensor, # [B, H, T * RANK_AB, K]
    b: torch.Tensor, # [B, H, T * RANK_AB, K]
    gk: torch.Tensor, # [B, H, T, K]
    RANK_AB: int,
    scale: Optional[int] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    head_first: bool = False,
    chunk_size: int = 32,
):
    o, final_state = chunk_hrdplr_delta_rule(
        q, # [B, H, T, K]
        k, # [B, H, T, K]
        v, # [B, H, T, K]
        a, # [B, H, 2 * T, K]
        b, # [B, H, 2 * T, K]
        gk, # [B, H, T, K]
        RANK_AB, 
        scale, 
        initial_state, 
        output_final_state, 
        head_first, 
        chunk_size, 
    )
    return o

if __name__ == '__main__':
    B = 64
    H = 16
    range_i = range(8, 12)
    K = 128   
    V = 128
    
    RANK_AB = 2

    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    scale = 1
    eps = 1e-6

    results = []  # 存储结果

    for i in range_i:
        T = 2 ** i
        print("Sequence length: ", T)
        initial_state = torch.randn(
            size=(B, H, K, V), device=device, dtype=dtype,
        )

        num_householder = 2
        q = torch.randn(
            size=(B, T, H, K), device=device, dtype=dtype, 
        )
        k = torch.randn(
            size=(B, T * num_householder, H, K), device=device, dtype=dtype, 
        )
        v = torch.randn(
            size=(B, T * num_householder, H, V), device=device, dtype=dtype, 
        )
        g = F.logsigmoid(
            torch.randn(
                size=(B, T, H), device=device, dtype=torch.float,
            )
        )
        beta = F.logsigmoid(
            torch.randn(
                size=(B, T * num_householder, H), device=device, dtype=torch.float,
            )
        )

        q = torch.repeat_interleave(
            q, dim=1, repeats=2, 
        )
        g = torch.repeat_interleave(g, repeats=2, dim=1)
        beta = torch.repeat_interleave(beta, repeats=2, dim=1)

        del q, k, v, g, beta

        # normalization
        q_t = torch.randn(
            size=(B, T * 2, H, K), device=device, dtype=dtype,
        )
        k_t = torch.randn(
            size=(B, T * 2, H, K), device=device, dtype=dtype,
        )
        v_t = torch.randn(
            size=(B, T * 2, H, K), device=device, dtype=dtype,
        )
        a_rank2_t = torch.randn(
            size=(B, T * 2, H, K), device=device, dtype=dtype,
        )
        b_rank2_t = torch.randn(
            size=(B, T * 2, H, K), device=device, dtype=dtype,
        )
        a_rank2_t /= (torch.norm(a_rank2_t, dim=-1, keepdim=True) + eps)
        b_rank2_t /= (torch.norm(b_rank2_t, dim=-1, keepdim=True) + eps)
        gk_t = F.logsigmoid(
            torch.randn(
                size=(B, T * 2, H, K), device=device, dtype=torch.float,
            )
        )
        gk_t = gk_t.transpose(-3, -2) # [B, T, H, K]
        gk_t = torch.cat([gk_t, torch.zeros_like(gk_t)], dim=1)
        gk_t = rearrange(gk_t, "b (c n) ... -> b (n c) ... ", c=2)
        gk_t = gk_t.contiguous()

        """
        Test the speed of naive implementation (through zero-padding, timestep x 2 and 
        calling existing diagonal-plus-rank-1 decay kernel in fla)
        """

        ms_dplr = triton.testing.do_bench(
            lambda: run_dplr(
                q_t,
                k_t,
                v_t,
                a_rank2_t,
                b_rank2_t,
                gk_t,
                scale,
                initial_state=initial_state,
                output_final_state=True,
                cu_seqlens=None,
                head_first=False,
            )
        )
        print(f"ms_dplr: {ms_dplr} ms")
        del q_t
        del k_t
        del v_t
        del a_rank2_t
        del b_rank2_t
        del gk_t

        q = torch.randn(
            size=(B, H, T, K), device=device, dtype=dtype,
        )

        k = torch.randn(
            size=(B, H, T, K), device=device, dtype=dtype,
        )
        v = torch.randn(
            size=(B, H, T, V), device=device, dtype=dtype,
        )
        a_rank2 = torch.randn(
            size=(B, H, T * RANK_AB, K), device=device, dtype=dtype,
        )

        b_rank2 = torch.randn(
            size=(B, H, T * RANK_AB, K), device=device, dtype=dtype,
        )
        initial_state = torch.randn(
            size=(B, H, K, V), device=device, dtype=dtype,
        )
        a_rank2 /= (torch.norm(a_rank2, dim=-1, keepdim=True) + eps)
        b_rank2 /= (torch.norm(b_rank2, dim=-1, keepdim=True) + eps)

        gk = F.logsigmoid(
            torch.randn(
                size=(B, H, T, K), device=device, dtype=torch.float,
            )
        )

        """
        Test the speed of customized kernel
        """

        ms_hrdplr_16 = triton.testing.do_bench(
            lambda: run_hrdplr(
                q,
                k,
                v,
                a_rank2,
                b_rank2,
                gk,
                RANK_AB,
                scale,
                initial_state,
                output_final_state=True,
                head_first=True,
                chunk_size=16,
            )
        )
        print(f"ms_hrdplr_16: {ms_hrdplr_16} ms")

        ms_hrdplr_32 = triton.testing.do_bench(
            lambda: run_hrdplr(
                q,
                k,
                v,
                a_rank2,
                b_rank2,
                gk,
                RANK_AB,
                scale,
                initial_state,
                output_final_state=True,
                head_first=True,
                chunk_size=32,
            )
        )
        print(f"ms_hrdplr_32: {ms_hrdplr_32} ms")

        ms_hrdplr_64 = triton.testing.do_bench(
            lambda: run_hrdplr(
                q,
                k,
                v,
                a_rank2,
                b_rank2,
                gk,
                RANK_AB,
                scale,
                initial_state,
                output_final_state=True,
                head_first=True,
                chunk_size=64,
            )
        )
        print(f"ms_hrdplr_64: {ms_hrdplr_64} ms")