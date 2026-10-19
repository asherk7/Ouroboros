"""RMSNorm for Ouroboros."""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    ``x / sqrt(mean(x**2) + eps) * weight``, with no bias or mean subtraction.
    The reduction runs in float32 so fp16 activations don't overflow.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        """Create the learned per-channel weight (init to ones).

        Args:
            dim: Size of the last axis.
            eps: Added inside the square root.
        """
        super().__init__()
        self.eps = eps
        # create the learned gain
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize over the last dim: ``(..., dim) -> (..., dim)``, same dtype as ``x``."""
        # upscale x to a higher type so that we don't risk overflow when computing the power
        # activations can overflow to infinity, which causes rsqrt to give 0, wiping out that activation position
        original_dtype = x.dtype
        x_upscale = x.to(torch.float32)
        # calculate the reciprocal square root so it's 1/RMS, use multiplication since it's faster
        inv_rms = torch.rsqrt(x_upscale.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        # cast the output back to the original dtype, since x_upscale & rms are fp32
        output = (x_upscale * inv_rms * self.weight).to(original_dtype)
        return output
