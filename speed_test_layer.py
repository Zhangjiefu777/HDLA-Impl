from HRDPLR import (
    HDLA_Custom, 
    HDLA_FLA
)

import torch
import torch.nn as nn
import torch.nn.functional as F

from fla.layers import (
    HGRN2Attention,
    GatedDeltaNet, 
    GatedDeltaProduct, 
    GatedLinearAttention, 
)

from .HRDPLR import (
    HDLA_Custom, 
    HDLA_FLA, 
)

if __name__ == '__main__':
    B = 2
    T = 256
    H = 4
    D = 128
    K = 64
    V = 64

    device = torch.device('cuda')
    dtype = torch.float32

    hdlr_custom_layer = HDLA_Custom(
        embed_dim=H * D, 
        num_heads=H, 
    )

    hrdplr_fla_layer = HDLA_FLA(
        embed_dim=H * D, 
        num_heads=H, 
    )