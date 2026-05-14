"""
DropPath or Stochastic Depth regularization
--------------

This regularization component is inspired by Stochastic Depth mentioned in the paper "Deep Networks with Stochastic
Depth" (https://arxiv.org/abs/1603.09382).

During training, randomly drops the entire residual branch for randomly selected samples in the batch with
probability ``drop_prob``.  Acts as a powerful regularizer and is distinct from standard Dropout (which drops
individual scalar activations). For a model with ``total_depth`` blocks, the drop probability is linearly
increased from 0 to ``drop_path_rate`` as block index increases:
        drop_prob_i = drop_path_rate x (i / (total_depth - 1))
"""

import torch
import torch.nn as nn
from torch import Tensor


class DropPath(nn.Module):
    """
    Stochastic Depth regularization (per-sample residual branch dropping).

    At training time, each sample in the batch has its residual contribution independently zeroed out with
    probability `drop_prob`.  The surviving contributions are rescaled by 1 / (1 - `drop_prob`) so the expected
    value is preserved — identical in spirit to standard Dropout but operating at the entire path level rather
    than individual activations. However, at inference time the module is the identity (pass-through).

    Args:
        drop_prob: Probability in [0, 1) that any given sample's path is dropped. 0.0 makes this an identity operation.

    Shape:
        - Input:  (B, ...) - any shape, operated on the batch dimension.
        - Output: same shape as input.

    Example:
        >>> dp = DropPath(drop_prob=0.1)
        >>> x = torch.ones(4, 256, 64)
        >>> out = dp(x)          # during training: some batch entries -> 0
        >>> out.shape
        torch.Size([4, 256, 64])
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        assert 0.0 <= drop_prob < 1.0, f"drop_prob must be in [0, 1), got {drop_prob}"
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply stochastic depth.

        Args:
            x: Input tensor of shape (B, *).

        Returns:
            Tensor of same shape as x.
        """
        if not self.training or self.drop_prob == 0.0:
            return x

        keep_prob = 1.0 - self.drop_prob
        B = x.shape[0]

        # Shape (B, 1, 1, ...) — one binary decision per sample, broadcast across all other dimensions so the whole
        # residual path is toggled.
        shape = (B,) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)   # Bernoulli

        # Scale surviving paths so E[output] = E[input].
        return x * random_tensor / keep_prob

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob:.4f}"
