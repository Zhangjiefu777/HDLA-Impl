import functools
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from packaging import version

if version.parse(torch.__version__) >= version.parse("2.4"):
    autocast_custom_fwd = functools.partial(torch.amp.custom_fwd, device_type="cuda")
    autocast_custom_bwd = functools.partial(torch.amp.custom_bwd, device_type="cuda")
else:
    autocast_custom_fwd = torch.cuda.amp.custom_fwd
    autocast_custom_bwd = torch.cuda.amp.custom_bwd

def contiguous(
    fn: Callable[..., torch.Tensor]
) -> Callable[..., torch.Tensor]:
    """
    A decorator to make sure all input tensors are contiguous.
    """
    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        return fn(ctx,
                  *(i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args),
                  **{k: (v if not isinstance(v, torch.Tensor) else v.contiguous()) for k, v in kwargs.items()})
    return wrapper

