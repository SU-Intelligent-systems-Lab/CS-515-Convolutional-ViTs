"""
Convolutional Token Embedding for CvT.

Each CvT stage begins by projecting its input (either the raw image or the
output feature map of the previous stage) into a sequence of `D`-dimensional
tokens via a single strided convolution followed by a `LayerNorm`.

This replaces the fixed-size, non-overlapping patch projection used in the
original Vision Transformer (ViT) with an overlapping, stride-based approach
that:

- Implicitly encodes local positional information through the convolution's receptive field — no explicit positional
  embeddings are needed.
- Progressively downsamples the spatial resolution across stages, giving the network a hierarchical, pyramid-like
  structure analogous to a CNN backbone (e.g. ResNet).
- Allows overlapping context between adjacent tokens when (kernel_size > stride) (the default in all three CvT stages).

Spatial resolution after embedding
-----------------------------------
For an input of spatial size `(H, W)`:

    H_out = floor((H + 2*padding - kernel_size) / stride) + 1
    W_out = floor((W + 2*padding - kernel_size) / stride) + 1

Default per-stage values (Tiny ImageNet, 64 x 64 input):

    Stage 1 (B,  3, 64, 64) [64x64 spatial] -> kernel=7, stride=4, pad=2 ->  (B, 256,  64) [16x16 spatial] (256 tokens)
    Stage 2 (B, 64, 16, 16) [16x16 spatial] -> kernel=3, stride=2, pad=1 ->  (B,  64, 192) [ 8x8  spatial] ( 64 tokens)
    Stage 3 (B,192,  8,  8) [ 8x8  spatial] -> kernel=3, stride=2, pad=1 ->  (B,  16, 384) [ 4x4  spatial] ( 16 tokens)

"""
import torch
import torch.nn as nn
from torch import Tensor


class ConvTokenEmbedding(nn.Module):
    """Convolutional Token Embedding used at the start of every CvT stage.

    Applies a single `Conv2d` (with learnable weights and optional bias) to the input feature map,
    then flattens the spatial dimensions into a sequence axis and normalizes with `LayerNorm`.

    The output is a 3-D tensor of shape `(B, N, C)` — the standard "batch-first" token format expected by all
    subsequent Transformer blocks — together with the 2-D spatial shape `(H_out, W_out)` so that downstream
    convolutions can reconstruct the feature map without carrying extra metadata.

    Args:
        in_channels: Number of channels in the input tensor (3 for the raw image; `embed_dim` of the previous stage
            otherwise).
        embed_dim: Embedding dimension `C` — the number of output channels produced by the convolution and the
            width of each token vector.
        kernel_size: Spatial size of the convolving kernel. Use 7 for Stage 1 (large receptive field on the raw
            image) and 3 for Stages 2–3.
        stride: Stride of the convolution. Controls the spatial downsampling ratio: `H_out = approx( H_in / stride )`.
        padding: Zero-padding added to both sides of the input. Setting `padding = kernel_size // 2` keeps the
            output size equal to `ceil(H_in / stride)` for odd kernel sizes.
        bias: If `True`, adds a learnable bias to the convolution.

    Shape:
        - Input:  `(B, in_channels, H, W)`
        - Output: `(B, H_out * W_out, embed_dim)`, `(H_out, W_out)`

    Example:
        >>> embed = ConvTokenEmbedding(
        ...     in_channels=3, embed_dim=64,
        ...     kernel_size=7, stride=4, padding=2,
        ... )
        >>> x = torch.randn(2, 3, 64, 64)
        >>> tokens, (h, w) = embed(x)
        >>> tokens.shape          # (2, 256, 64)
        torch.Size([2, 256, 64])
        >>> (h, w)                # spatial shape preserved for later conv ops
        (16, 16)
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        kernel_size: int,
        stride: int,
        padding: int,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Strided convolution: maps (B, C_in, H, W) -> (B, embed_dim, H', W').
        # A single Conv2d subsumes both the linear projection and the spatial
        # downsampling that ViT achieves with a patch-split + dense layer.
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

        # Normalize across the embedding dimension (last dim after reshape).
        # Applied after flattening so the norm sees the full token vectors.
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, tuple[int, int]]:
        """Embed the input feature map into a sequence of tokens.

        Steps:
            1. Conv2d               : `(B, C_in, H, W)`  -> `(B, C, H', W')`
            2. Flatten + transpose  : `(B, C, H', W')`   -> `(B, H'*W', C)`
            3. LayerNorm            : normalize along last dim `C`

        Args:
            x: Input feature map of shape `(B, in_channels, H, W)`.

        Returns:
            A tuple `(tokens, spatial_shape)` where

            * `tokens`        — `Tensor` of shape `(B, H' * W', embed_dim)`
            * `spatial_shape` — `(H_out, W_out)` as a plain Python tuple
              of ints, needed by downstream conv-attention layers to reshape
              the token sequence back into a 2-D feature map.
        """
        # 1. Project + downsample: (B, C_in, H, W) -> (B, embed_dim, H', W')
        x = self.proj(x)

        # Capture spatial dims before flattening — callers need them.
        h_out, w_out = x.shape[2], x.shape[3]

        # 2. Flatten spatial dims and move sequence axis to position 1:
        #    (B, embed_dim, H', W') -> (B, embed_dim, H'*W')
        #                           -> (B, H'*W', embed_dim)
        x = x.flatten(2).transpose(1, 2)

        # 3. Normalise token vectors.
        x = self.norm(x)

        return x, (h_out, w_out)

    # ---------------------------------------------------------------------- #
    # Helpers                                                                #
    # ---------------------------------------------------------------------- #

    def output_spatial_size(self, h_in: int, w_in: int) -> tuple[int, int]:
        """
        Compute the output spatial size for a given input size.

        Useful for pre-computing token counts without running a forward pass.
        Applies the standard convolution output-size formula:

            H_out = floor((H_in + 2*padding - kernel_size) / stride) + 1

        Args:
            h_in: Input height in pixels.
            w_in: Input width in pixels.

        Returns:
            `(H_out, W_out)` as a tuple of ints.

        Example:
            >>> embed = ConvTokenEmbedding(3, 64, kernel_size=7, stride=4, padding=2)
            >>> embed.output_spatial_size(64, 64)
            (16, 16)
        """
        h_out = (h_in + 2 * self.padding - self.kernel_size) // self.stride + 1
        w_out = (w_in + 2 * self.padding - self.kernel_size) // self.stride + 1
        return h_out, w_out

    def extra_repr(self) -> str:
        """Return a compact parameter summary shown in `print(model)`."""
        return (
            f"in_channels={self.in_channels}, "
            f"embed_dim={self.embed_dim}, "
            f"kernel_size={self.kernel_size}, "
            f"stride={self.stride}, "
            f"padding={self.padding}"
        )
