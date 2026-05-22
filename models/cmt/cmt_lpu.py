"""
CMT Local Perception Unit (LPU).

Position in the CMT architecture
----------------------------------
The LPU is the first submodule inside every CMT Block, running before the Lightweight Multi-Head Self-Attention (LMHSA)
and the Inverted Residual Feed-Forward Network (IRFFN).

From the CMT paper (Section 3.2), LPU(X_i) = DW-Conv(X_i) + X_i, where DW-Conv denotes a depthwise convolution with a 
3x3 kernel. This is the complete definition - no normalization, no nonlinear activation, and no additional learned 
parameters beyond the depthwise conv kernel and bias.


Why LPU exists
--------------
Pure self-attention is permutation-equivariant: shuffling all tokens produces identically shuffled outputs, so the 
model has no inherent sense of spatial locality.
Rather than adding explicit positional embeddings (as ViT does with sinusoidal or learnable vectors), CMT injects 
local spatial structure through the LPU's 3x3 depthwise convolution, which:

1.  Aggregates information from each token's 8 immediate spatial neighbors.
2.  Encodes relative position implicitly via the kernel's local receptive field.
3.  Adds zero positional embedding parameters on top of the conv itself, position is encoded by the weight geometry, 
    not separate embedding tables.

Implementation note
-------------------
The LPU receives a token sequence (B, N, C) (standard Transformer layout), but `Conv2d` requires a spatial 
tensor (B, C, H, W). Two view operations (transpose + reshape, and their inverses) bracket the convolution. 
The spatial shape (H, W) must be passed at every forward call so the module knows how to arrange the (N = H x W) tokens.
"""
import torch.nn as nn
from torch import Tensor


class CMT_LPU(nn.Module):
    """
    Local Perception Unit: residual 3x3 depthwise convolution.
    Implements the formula from the CMT paper exactly: LPU(X) = DW-Conv(X) + X

    A depthwise 3x3 convolution is applied to the 2-D spatial layout of the token sequence, and the result is added 
    back to the input as a residual. No normalization and no activation are applied, the LPU is intentionally
    minimal so as not to disturb the raw spatial statistics it is designed to encode.
    
    Depthwise convolution means `groups = channels`: each channel is convolved by its own 3x3 kernel independently, 
    keeping the operation parameter-efficient (only C x 3 x 3 kernel parameters, no cross-channel mixing).

    Args:
        dim: Token channel dimension C. Must match the embedding dimension of the current CMT stage.

    Shape:
        - Input:  (B, N, C)  where N = H x W
        - Output: (B, N, C) - same shape, local context added

    Example:
        >>> import torch
        >>> lpu = CMT_LPU(dim=46)
        >>> x = torch.randn(2, 256, 46)   # stage 1: 16x16 grid, 46-dim tokens
        >>> out = lpu(x, H=16, W=16)
        >>> out.shape
        torch.Size([2, 256, 46])
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

        # Depthwise 3x3 convolution (groups=dim -> one kernel per channel).
        # kernel=3, stride=1, padding=1 -> spatial size unchanged.
        # bias=True: no normalization (BN or LN) follows the conv, so the bias is meaningful.
        self.dw_conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1,
                                 groups=dim, bias=True)

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        """
        Apply the Local Perception Unit.

        Steps:
            1. Reshape (B, N, C) -> (B, C, H, W)
            2. Apply depthwise 3x3 conv ->  (B, C, H, W)
            3. Reshape back (B, C, H, W) -> (B, N, C)
            4. Add residual ->  x + DW-Conv(x)

        Args:
            x: Token sequence (B, N, C) where N = H x W.
            H: Spatial height of the current feature map (needed for reshape).
            W: Spatial width of the current feature map (needed for reshape).

        Returns:
            Locally-enriched token sequence (B, N, C).
        """
        B, N, C = x.shape

        # 1. Sequence -> spatial feature map
        # (B, N, C) -> (B, C, H, W)
        feat = x.permute(0, 2, 1).reshape(B, C, H, W)

        # 2. Depthwise 3x3 convolution.
        # Each channel is filtered by its own independent 3x3 kernel.
        # Padding=1 keeps the spatial size (H, W) unchanged.
        feat = self.dw_conv(feat)                        # (B, C, H, W)

        # 3. Spatial feature map -> sequence
        # (B, C, H, W) -> (B, C, N) -> (B, N, C)
        feat = feat.flatten(2).permute(0, 2, 1)

        # 4. Residual addition: LPU(X) = DW-Conv(X) + X
        # The residual lets the block learn an incremental local correction on top of what the token already represents,
        # rather than replacing it.
        return x + feat

    def extra_repr(self) -> str:
        """Compact summary shown in `print(model)`."""
        params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return f"dim={self.dim}, kernel=3x3, depthwise (groups=dim), params={params}"
    