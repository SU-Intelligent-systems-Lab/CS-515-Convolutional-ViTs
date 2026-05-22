"""
CMT Inverted Residual Feed-Forward Network (IRFFN).

Position in the CMT architecture
----------------------------------
The `IRFFN` replaces the standard Feed-Forward Network (FFN/MLP) typically found in Vision Transformers. It serves as
the final core operational block within each CMT Block stack, operating immediately after the Local Position Unit (LPU)
and Lightweight Multi-Head Self-Attention (LMHSA) stages.

Design rationale
-----------------
While standard ViT MLPs only use 1x1 token-wise projections, CMT's IRFFN infuses architectural mechanisms heavily
inspired by MobileNetV2's inverted residuals to maximize spatial correlation:

1.  Expansion (1x1 Conv): Projects input tokens into a much higher dimensional subspace (controlled by `mlp_ratio`,
    typically expanding channels by 4x) using `conv1`.

2.  Depthwise Convolution (3x3 DWConv): Introduces a lightweight spatial structural inductive bias via `proj`.
    Crucially, a residual shortcut (`+ x`) wraps around this depthwise layer to safeguard gradient flow and retain
    fine-grained feature details.

3.  Compression (1x1 Conv): Projects the expanded, spatially aggregated representation back down to the target
    architecture embedding space using `conv2`.

Because these operations require 2D spatial layouts, the module shifts dynamically between sequence
(B, N, C) and feature-map (B, C, H, W) formats inside its forward pass.
"""
import torch.nn as nn
from torch import Tensor


class CMT_IRFFN(nn.Module):
    """
    Inverted Residual Feed-Forward Network (IRFFN) block.

    Accepts a 1D token sequence of shape (B, N, C), reshapes it into a 2D spatial feature map
    (B, C, H, W) to execute a sequence of channel expansions, localized depthwise convolutions,
    and channel compressions, before flattening back to a 1D token layout.

    Args:
        dim: Number of input/output embedding channels.
        mlp_ratio: Expansion factor determining the inner hidden dimension size. Default: 4.0.
        drop: Dropout probability applied to final activations. Default: 0.0.

    Shape:
        - Input:  (B, N, C) where N = H * W
        - Output: (B, N, C)

    Example:
        >>> import torch
        >>> layer = CMT_IRFFN(dim=46, mlp_ratio=4.0, drop=0.1)
        >>> x = torch.randn(2, 3136, 46)
        >>> out = layer(x, H=56, W=56)
        >>> out.shape
        torch.Size([2, 3136, 46])
    """

    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0) -> None:
        super().__init__()

        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.hidden = int(dim * mlp_ratio)      # Expanded hidden channel count (C_h)

        # 1. Expand channels: Pointwise linear projection layer
        self.conv1 = nn.Conv2d(in_channels=self.dim, out_channels=self.hidden, kernel_size=1, stride=1,
                               padding=0, bias=True)

        # 2. Local spatial interaction: Depthwise spatial convolution layer
        self.proj = nn.Conv2d(in_channels=self.hidden, out_channels=self.hidden, kernel_size=3, stride=1,
                              padding=1, groups=self.hidden)

        # 3. Compress channels: Pointwise linear reduction layer
        self.conv2 = nn.Conv2d(in_channels=self.hidden, out_channels=self.dim, kernel_size=1, stride=1,
                               padding=0, bias=True)

        # Non-linear activations, normalizations, and regularization
        self.act = nn.GELU()
        self.bn = nn.BatchNorm2d(self.hidden)
        self.final_bn = nn.BatchNorm2d(self.dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        """
        Execute the inverted residual forward pass.

        Steps:
            1. Transpose token sequence (B, N, C) to spatial feature maps (B, C, H, W).
            2. Apply initial 1x1 point-wise expansion + GELU + BatchNorm.
            3. Apply 3x3 depthwise convolution with a local skip connection.
            4. Apply second GELU + BatchNorm activation step.
            5. Compress features back down using a final 1x1 projection + BatchNorm + Dropout.
            6. Re-flatten back into token sequence formatting (B, N, C).

        Args:
            x: Input token sequence of shape (B, N, C).
            H: Target reconstitution height.
            W: Target reconstitution width.

        Returns:
            Processed tensor sequence of shape (B, N, C).
        """
        B, N, C = x.shape

        # 1. Sequence to Feature Map Transformation
        # (B, N, C) -> (B, C, N) -> (B, C, H, W)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)

        # 2. Linear Channel Expansion
        # (B, C, H, W) -> (B, C_h, H, W)
        x = self.conv1(x)
        x = self.act(x)
        x = self.bn(x)

        # 3. Depthwise Convolution + Inverted Local Residual
        # (B, C_h, H, W) + (B, C_h, H, W) -> (B, C_h, H, W)
        x = self.proj(x) + x
        x = self.act(x)
        x = self.bn(x)

        # 4. Linear Channel Compression & Normalization
        # (B, C_h, H, W) -> (B, C, H, W)
        x = self.conv2(x)
        x = self.final_bn(x)
        x = self.drop(x)

        # 5. Feature Map back to Sequence Transformation
        # (B, C, H, W) -> (B, C, N) -> (B, N, C)
        x = x.flatten(2).permute(0, 2, 1)

        return x

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters in this IRFFN."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extra_repr(self) -> str:
        """Compact summary shown in `print(model)`."""
        return (
            f"dim={self.dim}, hidden={self.hidden}, "
            f"mlp_ratio={self.mlp_ratio}, params={self.num_parameters:,}"
        )
