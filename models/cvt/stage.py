"""
CvT Stage: assembles ConvTokenEmbedding + a stack of CvT Blocks.

A CvTStage is the top-level building block that is repeated three times inside the full CvT model.

Each stage is responsible for:

1.  Spatial downsampling via ConvTokenEmbedding: takes the raw image (Stage 1) or the reshaped output of the previous
    stage (Stages 2–3) and produces a new, spatially smaller token sequence with a larger embedding dimension.

2.  Contextual refinement via a stack of CvT Blocks: runs the token sequence through `depth` CvT Transformer Blocks,
    each of which applies ConvAttention + FFN with pre-LayerNorm and residual connections.

3.  Stochastic depth scheduling: receives a slice of the global linearly-spaced drop-probability list, one value per
    block, and wires each CvTBlock with its assigned drop probability. This means the stage does not need to know its
    absolute position in the full network; the parent `CvT` module computes the global schedule and distributes slices.

4.  Spatial shape propagation: `ConvTokenEmbedding.forward()` returns the output spatial shape (H_out, W_out). The
    stage stores this and threads it through every block (which needs it to reshape the token sequence back to 2-D
    for convolutional Q/K/V projections). The final (H, W) is returned to the parent so the next stage can reshape
    correctly.


Data flow
---------
    Input: (B, C_in, H_in, W_in) - image tensor or previous-stage feature map
         ↓
    ConvTokenEmbedding
         │  (B, N, embed_dim)  +  (H_out, W_out)
         │
    CvTBlock 0   (drop_path_probs[0])
    CvTBlock 1   (drop_path_probs[1])
    ...
    CvTBlock d-1 (drop_path_probs[d-1])
         ↓
    Output: (B, N, embed_dim),  (H_out, W_out)

The output tokens can be:
- Reshaped to (B, embed_dim, H_out, W_out) and fed to the next stage's `ConvTokenEmbedding` (Stages 1 -> 2 and 2 -> 3).
- Kept as (B, N, embed_dim) for the CLS token + MLP head (Stage 3).
"""
import torch.nn as nn
from torch import Tensor
from models.cvt.conv_embed import ConvTokenEmbedding
from models.cvt.cvt_block import CvTBlock


class CvTStage(nn.Module):
    """
    One stage of the CvT hierarchical pipeline.

    Combines a `ConvTokenEmbedding` (spatial downsampling + projection) with a sequential stack of `CvTBlock` modules
    (convolutional self-attention + FFN). The stage is completely self-contained: it manages its own token embedding,
    block stack, and stochastic depth wiring.

    Args:
        in_channels: Number of channels in the input feature map.  For Stage 1 this is the image channel count
                     (typically 3); for later stages it equals the `embed_dim` of the previous stage.
        embed_dim: Embedding dimension for this stage (output channels of `ConvTokenEmbedding` and width of each
                   token vector).
        depth: Number of `CvTBlock` modules to stack.
        num_heads: Number of attention heads in each `CvTBlock`.
        kernel_size_embed: Kernel size for the `ConvTokenEmbedding` conv.
        stride_embed: Stride for the `ConvTokenEmbedding` conv.
        padding_embed: Padding for the `ConvTokenEmbedding` conv.
        kernel_size_proj: Kernel size for convolutional Q/K/V projections inside each `CvTBlock`.
        stride_kv: Stride for K and V projections inside each `CvTBlock`.
        mlp_ratio: FFN hidden-dimension expansion factor.
        drop_rate: Dropout probability inside the FFN of each block.
        attn_drop_rate: Dropout probability on attention weights.
        drop_path_probs: A list of per-block stochastic-depth drop probabilities of length `depth`. Expected to be a
                         monotonically non-decreasing slice of the global linear schedule produced by `CvT`.  If None,
                         all blocks use `drop_path=0`.
        qkv_bias: Whether to add learnable bias to Q/K/V conv projections.

    Shape:
        - Input: (B, in_channels, H_in, W_in)
        - Output: (B, H_out * W_out, embed_dim), (H_out, W_out)

    Example:
        >>> stage = CvTStage(
        ...     in_channels=3,   embed_dim=64,  depth=1, num_heads=1,
        ...     kernel_size_embed=7, stride_embed=4, padding_embed=2,
        ...     kernel_size_proj=3,  stride_kv=2,
        ...     drop_path_probs=[0.0],
        ... )
        >>> x = torch.randn(2, 3, 64, 64)
        >>> tokens, (h, w) = stage(x)
        >>> tokens.shape          # (2, 256, 64)
        torch.Size([2, 256, 64])
        >>> (h, w)                # spatial dims for next stage / attention
        (16, 16)
    """

    def __init__(self, in_channels: int, embed_dim: int, depth: int, num_heads: int, kernel_size_embed: int,
                 stride_embed: int, padding_embed: int, kernel_size_proj: int, stride_kv: int, mlp_ratio: float = 4.0,
                 drop_rate: float = 0.0, attn_drop_rate: float = 0.0, drop_path_probs: list[float] | None = None,
                 qkv_bias: bool = True) -> None:
        super().__init__()

        if drop_path_probs is None:
            drop_path_probs = [0.0] * depth

        if len(drop_path_probs) != depth:
            raise ValueError(f"drop_path_probs has length {len(drop_path_probs)} but depth={depth}. They must match!")

        # ----------- 1. Convolutional Token Embedding -----------
        self.embed = ConvTokenEmbedding(
            in_channels=in_channels,
            embed_dim=embed_dim,
            kernel_size=kernel_size_embed,
            stride=stride_embed,
            padding=padding_embed,
        )

        # ---------------- 2. Stack of CvT Blocks ----------------
        # Each block receives its own drop probability from the schedule
        # slice so stochastic depth increases with block depth.
        self.blocks = nn.Sequential(
            *[
                CvTBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    kernel_size=kernel_size_proj,
                    stride_kv=stride_kv,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path_probs[i],
                    qkv_bias=qkv_bias,
                )
                for i in range(depth)
            ]
        )

        # Store for extra_repr
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads

    def forward(self, x: Tensor) -> tuple[Tensor, tuple[int, int]]:
        """
        Run the stage: embed -> refine through blocks -> return tokens.

        Args:
            x:  Input tensor of shape (B, in_channels, H_in, W_in). For Stage 1 this is the raw image batch.
                For Stages 2–3 it is the previous stage's token sequence reshaped back to a 2-D feature map by the
                parent `CvT` module.

        Returns:
            A tuple (tokens, spatial_shape) where:
            - `tokens`        — (B, H_out * W_out, embed_dim)
            - `spatial_shape` — (H_out, W_out) needed by the next stage and by the MLP head in `CvT`.
        """
        # -------- Token embedding: image/feature-map -> sequence --------
        # tokens : (B, N, embed_dim) where N = H_out x W_out
        # h, w : output spatial dims (needed by ConvAttention inside blocks)
        tokens, (h, w) = self.embed(x)

        # ----------- CvT Blocks: refine token representations -----------
        # nn.Sequential cannot forward extra args, so we iterate manually. Each block needs (tokens, h, w) because
        # ConvAttention reshapes the sequence back to (B, C, H, W) for its depthwise conv projections.
        for block in self.blocks:
            tokens = block(tokens, h, w)

        return tokens, (h, w)

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    def output_spatial_size(self, h_in: int, w_in: int) -> tuple[int, int]:
        """
        Compute the output spatial size without running a forward pass.

        Delegates to `ConvTokenEmbedding.output_spatial_size`. Useful for
        pre-computing token counts, allocating buffers, or verifying config.

        Args:
            h_in: Input spatial height.
            w_in: Input spatial width.

        Returns:
            (H_out, W_out) after this stage's `ConvTokenEmbedding`.

        Example:
            >>> stage.output_spatial_size(64, 64)
            (16, 16)
        """
        return self.embed.output_spatial_size(h_in, w_in)

    def extra_repr(self) -> str:
        """Compact parameter summary shown in ``print(model)``."""
        return f"embed_dim={self.embed_dim}, depth={self.depth}, num_heads={self.num_heads}"
    