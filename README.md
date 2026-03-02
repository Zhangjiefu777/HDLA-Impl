### Official Kernel Implementation of Householder Diagonalized Linear Attention (HDLA)

This repository contains the official implementation of ICLR 2026 poster paper: Householder-Diagonalized Linear Attention (HDLA): Utilizing Enhanced Decay Mechanism for Efficient Sequence Modeling (paper link: https://openreview.net/forum?id=HVFjzaQeig).

The kernel implementation is rank-generalized (with $r_{ab} = 2, r_{kv} = 1$) from the generalized delta rule (with $r_{ab} = 1, r_{kv} = 1$) kernel[1] in FlashLinearAttention.

#### References
[1] https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/generalized_delta_rule/dplr
