"""
CMT Patch Embedding.

Position in the CMT architecture
----------------------------------
A `PatchEmbedding` layer sits at the entry of every CMT stage, immediately before the stage's stack of CMT Blocks.
There are four instances in total (one per stage). From Table 1 of the CMT paper, every Patch Embedding layer uses
the same configuration: `kernel_size=2`, `stride=2`, `padding=0` for all four stages.

Design rationale
-----------------
The 2x2 non-overlapping convolution acts as a learned patch-splitting operation, each output token aggregates exactly
one 2x2 window of the input feature map. The simultaneous stride-2 downsampling halves the spatial resolution and the
output channel count is projected to the next stage's embedding dimension via the convolution weight matrix.

`bias=False` because the following `LayerNorm` provides its own learnable affine shift (γ, β), making a bias term on
the convolution redundant.

The output format is a token sequence (B, N, C) plus the spatial shape (H_out, W_out), which is the exact format
expected by CMT Blocks. The spatial shape must be carried explicitly so that the depthwise convolutions inside LPU,
LMHSA's spatial reduction, and IRFFN can reconstruct the 2-D layout.
"""
import torch.nn as nn
from torch import Tensor


class PatchEmbedding(nn.Module):
    """
    Patch Embedding layer for CMT- 2x2 strided Conv2d + LayerNorm.

    Accepts a spatial feature map (B, in_dim, H, W) and produces a token sequence (B, N_out, out_dim) at half spatial
    resolution. All four PatchEmbedding instances in a CMT model use identical `kernel_size=2` and `stride=2` as
    specified in Table 1 of the CMT paper.

    Args:
        in_dim: Number of input channels. Equal to `stem_channels` for the first stage; equal to `channel_dims[i-1]`
                for subsequent stages.
        out_dim: Number of output channels - the embedding dimension for the current stage (`cmt_channel_dims[i]` from
                 `ModelConfig`).

    Shape:
        - Input:  (B, in_dim, H, W)
        - Output: (B, H/2 * W/2, out_dim),  (H/2, W/2)

    Example:
        >>> pe = PatchEmbedding(in_dim=16, out_dim=46)
        >>> import torch
        >>> x = torch.randn(2, 16, 32, 32)   # stem output for CMT-Ti
        >>> tokens, (h, w) = pe(x)
        >>> tokens.shape
        torch.Size([2, 256, 46])
        >>> (h, w)
        (16, 16)
    """
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        # 2x2 non-overlapping projection: each output position corresponds to exactly one 2x2 patch of the
        # input (kernel_size == stride, padding=0).
        # Simultaneously performs spatial downsampling and channel projection.
        # bias=False: LayerNorm provides its own learnable affine (γ, β).
        self.proj = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=2, stride=2, padding=0, bias=False)

        # Normalize over the channel dimension of the flattened token sequence.
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, tuple[int, int]]:
        """
        Project and downsample the input feature map into a token sequence.

        Steps:
            1. `Conv2d(k=2, s=2)`: simultaneous 2x spatial downsampling and channel projection:
                                   (B, in_dim, H, W) -> (B, out_dim, H/2, W/2)
            2. Flatten spatial dims and transpose to sequence format: (B, out_dim, H/2, W/2) -> (B, H/2 * W/2, out_dim)
            3. `LayerNorm` over the channel (last) dimension.

        Args:
            x: Spatial feature map of shape (B, in_dim, H, W).

        Returns:
            A tuple (tokens, spatial_shape) where:

            - `tokens`: Tensor of shape (B, H/2 * W/2, out_dim)
            - `spatial_shape`: (H_out, W_out) carried to every CMT Block that contains a DWConv operation.
        """
        # 1. 2x2 conv: spatial downsample + channel project
        # (B, in_dim, H, W) -> (B, out_dim, H/2, W/2)
        x = self.proj(x)
        H_out, W_out = x.shape[2], x.shape[3]

        # 2. Feature map -> token sequence
        # (B, out_dim, H', W') -> (B, out_dim, N) -> (B, N, out_dim)
        x = x.flatten(2).transpose(1, 2)

        # 3. LayerNorm over channel dimension
        x = self.norm(x)

        return x, (H_out, W_out)

    def output_spatial_size(self, h_in: int, w_in: int) -> tuple[int, int]:
        """
        Compute the output spatial size without running a forward pass.

        With `kernel_size=stride=2` and `padding=0` the formula simplifies to: `H_out = H_in / 2`

        Args:
            h_in: Input spatial height.
            w_in: Input spatial width.

        Returns:
            (H_out, W_out) after this embedding layer.

        Example:
            >>> pe = PatchEmbedding(16, 46)
            >>> pe.output_spatial_size(32, 32)
            (16, 16)
        """
        return h_in // 2, w_in // 2

    def extra_repr(self) -> str:
        """Compact summary shown in `print(model)`."""
        return (
            f"in_dim={self.in_dim}, out_dim={self.out_dim}, kernel=2x2, stride=2 (non-overlapping, see paper Table 1)"
        )
