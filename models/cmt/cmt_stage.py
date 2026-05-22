"""
CMT Stage - one hierarchical stage of the CMT model.

Position in the CMT architecture
----------------------------------
There are four `CMTStage` instances in a complete CMT model. Each stage owns a PatchEmbedding layer
(spatial downsampling + channel projection) followed by a stack of CMTBlock modules.

Stage responsibilities
----------------------
1.  PatchEmbedding: converts the incoming 2-D feature map to a token sequence at half spatial resolution
    (kernel=2, stride=2, fixed for all stages per Table 1).

2.  DropPath scheduling: receives a slice of the global linearly-spaced drop probability list. Each CMTBlock is
    wired with its assigned probability. The stage does not need to know its absolute position in the full network;
    the parent `CMT` handles that.

3.  Spatial shape propagation: `PatchEmbedding.forward` returns (H_out, W_out). The stage stores these and threads
    them through every block, which needs the spatial layout for the DWConv operations inside LPU, LMHSA's spatial
    reduction, and IRFFN.

Note: CMTStage returns (tokens, (H, W)) — a token sequence, not a feature map. The parent CMT model is responsible
for reshaping back to (B, C, H, W) between stages so the next PatchEmbedding can accept a 2-D spatial input.

Relationship to CvT's `CvTStage`
------------------------------------
The two stage classes are structurally parallel:

    CvTStage: ConvTokenEmbedding -> stack of CvTBlocks -> return (tokens, (H, W))
    CMTStage: PatchEmbedding -> stack of CMTBlocks -> return (tokens, (H, W))

The key differences:
-   `PatchEmbedding` (non-overlapping conv) vs `ConvTokenEmbedding` (overlapping).
-   `CMTBlock` (LPU + LMHSA + IRFFN) vs `CvTBlock` (ConvAttention + FFN).
-   The `nn.Sequential` trick cannot be used for `CMTBlock` (as in `CvTStage`) because CMT blocks require
    (x, H, W) three arguments. We iterate manually, same as CvT.
"""

import torch.nn as nn
from torch import Tensor
from models.cmt.cmt_patch_embed import PatchEmbedding
from models.cmt.cmt_block import CMTBlock


class CMTStage(nn.Module):
    """
    One stage of the CMT hierarchical pipeline.

    Combines a `PatchEmbedding` layer with a sequential stack of `CMTBlock` modules (LPU + LMHSA + IRFFN).

    Args:
        in_dim: Input channel count. Equals `stem_channels` for stage 1; `cmt_channel_dims[i-1]` for stages 2–4.
        embed_dim: Output embedding dimension for this stage (`cmt_channel_dims[i]` from `ModelConfig`).
        depth: Number of `CMTBlock` modules to stack (`cmt_depths[i]` from `ModelConfig`).
        num_heads: Attention heads per block. Fixed per stage across all variants: `(1, 2, 4, 8)` for stages 1–4.
        sr_ratio: Spatial reduction ratio for K/V in LMHSA. Passed unchanged to every block (all blocks in a stage
                  share the same ratio). Fixed per stage: `(8, 4, 2, 1)` for stages 1–4.
        mlp_ratio: IRFFN hidden-dim expansion factor for this stage (`cmt_mlp_ratios[i]` from `ModelConfig`).
        qkv_bias: Learnable bias in LMHSA projections.
        drop_rate: Dropout probability for IRFFN and LMHSA output projections.
        attn_drop_rate: Dropout probability on attention weights.
        drop_path_probs: Per-block stochastic-depth drop probabilities, length `depth`. Expected to be a monotonically
                         non-decreasing slice of the global linear schedule produced by `CMT`. Defaults to all-zero if
                         `None`.

    Shape:
        - Input:  `(B, in_dim, H_in, W_in)`   <- 2-D feature map
        - Output: `(B, H_out*W_out, embed_dim)`,  `(H_out = H_in // 2, W_out = W_in // 2)`

    Example:
        >>> stage = CMTStage(
        ...     in_dim=32, embed_dim=46, depth=2, num_heads=1,
        ...     sr_ratio=8, pa_stride=4, mlp_ratio=3.6,
        ...     drop_path_probs=[0.0, 0.01],
        ... )
        >>> import torch
        >>> x = torch.randn(2, 32, 32, 32)   # stem output, CMT-Ti
        >>> tokens, (h, w) = stage(x)
        >>> tokens.shape
        torch.Size([2, 64, 46])
        >>> (h, w)
        (8, 8)
    """

    def __init__(self, in_dim: int, embed_dim: int, depth: int, num_heads: int, sr_ratio: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, drop_rate: float = 0.0, attn_drop_rate: float = 0.0,
                 drop_path_probs: list[float] | None = None) -> None:
        super().__init__()

        if drop_path_probs is None:
            drop_path_probs = [0.0] * depth

        if len(drop_path_probs) != depth:
            raise ValueError(f"drop_path_probs has {len(drop_path_probs)} entries but depth={depth}. "
                             f"Lengths must match.")

        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.sr_ratio = sr_ratio

        # 1. Patch Embedding
        # Converts the incoming feature map (B, in_dim, H, W) to a token sequence (B, N_out, embed_dim) at reduced
        # spatial resolution (kernel=2, stride=2 for all stages).
        self.patch_embed = PatchEmbedding(in_dim=in_dim, out_dim=embed_dim)

        # 2. Stack of CMT Blocks
        # Each block receives a monotonically increasing drop probability from the global DropPath schedule.
        # nn.ModuleList (not Sequential) because CMTBlock.forward needs three arguments: (x, H, W).
        # Spatial size is constant within the stage, only PatchEmbedding downsamples.
        self.blocks = nn.ModuleList([
            CMTBlock(
                dim=embed_dim,
                num_heads=num_heads,
                sr_ratio=sr_ratio,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_probs[i],
            )
            for i in range(depth)
        ])

    def forward(self, x: Tensor) -> tuple[Tensor, tuple[int, int]]:
        """
        Run one CMT stage: PatchEmbedding -> CMTBlock stack.

        Args:
            x:  Input feature map (B, in_dim, H_in, W_in). For stage 1 this is the stem output; for stages 2–4 it is
                the previous stage's token sequence reshaped back to 2-D by the parent CMT model.

        Returns:
            A tuple (tokens, spatial_shape) where:
            - `tokens`: (B, H_out * W_out, channel_dim)
            - `spatial_shape`: (H_out, W_out) needed by the next stage and by CMT's `forward` for the feature-map
                               reshape between stages.
        """
        # PatchEmbedding: feature map (B, in_dim, H, W) -> token sequence (B, N, channel_dim),  (H_out, W_out)
        tokens, (H, W) = self.patch_embed(x)

        # CMTBlocks: (B, N, C) → (B, N, C), spatial (H, W) unchanged
        for block in self.blocks:
            tokens = block(tokens, H, W)

        return tokens, (H, W)

    def output_spatial_size(self, h_in: int, w_in: int) -> tuple[int, int]:
        """
        Compute the output spatial size without running a forward pass.

        Delegates to `PatchEmbedding.output_spatial_size` (H / 2, W / 2).

        Args:
            h_in: Input spatial height.
            w_in: Input spatial width.

        Returns:
            `(H_out, W_out)` after this stage's `PatchEmbedding`.

        Example:
            >>> stage.output_spatial_size(32, 32)
            (16, 16)
        """
        return self.patch_embed.output_spatial_size(h_in, w_in)

    def extra_repr(self) -> str:
        """Compact summary shown in `print(model)`."""
        return f"channel_dim={self.channel_dim}, depth={self.depth}, num_heads={self.num_heads}, sr_ratio={self.sr_ratio}"
