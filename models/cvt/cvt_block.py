"""
Convolutional Transformer Block
--------------
A single Convolutional Transformer Block is the repeating unit stacked ``depth`` times inside every stage.
It follows the canonical **pre-norm** Transformer design:

        X
 -------│
│       ↓
│   LayerNorm
│   ConvAttention
│   DropPath
│--> Residual (+)
        ↓
        X
 -------│
│       ↓
│   LayerNorm
│   FFN (MLP)
│   DropPath
│--> Residual (+)
        ↓
        X


Key components
--------------
- Pre-LayerNorm: Normalization is applied before each sub-module, not after. This leads to more stable training.

- ConvAttention: Replaces the standard linear Q/K/V projection with depthwise separable convolutions.

- FFN (Feed-Forward Network): a two-layer MLP that projects each token independently: 
        Linear(C, C x r) -> GELU -> Dropout -> Linear(C x r, C) -> Dropout.
        (No spatial operations. Purely per-token channel mixing.)

- DropPath (Stochastic Depth, See "Deep Networks with Stochastic Depth" in https://arxiv.org/abs/1603.09382):
    During training, randomly drops the entire residual branch for randomly selected samples in the batch with
    probability ``drop_prob``.  Acts as a powerful regularizer and is distinct from standard Dropout (which
    drops individual scalar activations). For a model with ``total_depth`` blocks, the drop probability is linearly
    increased from 0 to ``drop_path_rate`` as block index increases:
        drop_prob_i = drop_path_rate x (i / (total_depth - 1))
    This schedule is computed in CvTStage and passed per-block, so each block need not know its absolute position.
"""

import torch
import torch.nn as nn
from torch import Tensor
from models.cvt.conv_attention import ConvAttention
from models.cvt.drop_path import DropPath
from models.cvt.cvt_mlp import CvTFFN


class CvTBlock(nn.Module):
    """
    One CvT Transformer Block (pre-norm, conv-attention, stochastic depth).

    Stacks two submodules with residual connections and pre-normalization:

    1. Attention branch: LN -> ConvAttention -> DropPath -> + (residual)
    2. FFN branch: LN -> FFN -> DropPath -> + (residual)

    The spatial shape (H, W) must be passed at every forward call because ConvAttention needs it to reshape the token
    sequence into a 2-D feature map for its convolutional Q/K/V projections.

    Args:
        embed_dim: Token embedding dimension C.
        num_heads: Number of attention heads.
        mlp_ratio: FFN hidden-dim expansion factor.
        kernel_size: Kernel size for convolutional Q/K/V projections.
        stride_kv: Stride for K and V projections (spatial downsampling).
        drop: Dropout probability for FFN layers.
        attn_drop: Dropout probability on attention weights.
        drop_path: Stochastic depth drop probability for this block. Should increase linearly with block index
                   (scheduled in Stage).
        qkv_bias: Learnable bias in Q/K/V convolutional projections.

    Shape:
        - Input:  (B, N, C) + (H, W)  where N = H x W
        - Output: (B, N, C)

    Example:
        >>> block = CvTBlock(
        ...     embed_dim=64, num_heads=1, mlp_ratio=4.0,
        ...     kernel_size=3, stride_kv=2,
        ... )
        >>> x = torch.randn(2, 256, 64)
        >>> out = block(x, h=16, w=16)
        >>> out.shape
        torch.Size([2, 256, 64])
    """
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, kernel_size: int = 3,
                 stride_kv: int = 2, drop: float = 0.0, attn_drop: float = 0.0, drop_path: float = 0.0,
                 qkv_bias: bool = True) -> None:
        super().__init__()

        # Pre-norm before attention
        self.norm1 = nn.LayerNorm(embed_dim)

        # Convolutional multi-head self-attention
        self.attn = ConvAttention(embed_dim=embed_dim, num_heads=num_heads, kernel_size=kernel_size,
                                  stride_kv=stride_kv, attn_drop=attn_drop, proj_drop=drop, qkv_bias=qkv_bias)

        # Pre-norm before FFN
        self.norm2 = nn.LayerNorm(embed_dim)

        # Position-wise feed-forward network
        self.ffn = CvTFFN(in_features=embed_dim, mlp_ratio=mlp_ratio, drop=drop)

        # Stochastic depth (identity when drop_path == 0)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: Tensor, h: int, w: int) -> Tensor:
        """
        Apply one CvT Transformer Block.

        Args:
            x: Token sequence (B, N, C) where N = H x W.
            h: Spatial height of the current feature map.
            w: Spatial width of the current feature map.

        Returns:
            Transformed token sequence (B, N, C) - same shape as input, ready for the next block or the next
            stage's ConvTokenEmbedding.
        """
        # ------------ Attention branch ------------
        # x = x + DropPath( ConvAttention( LayerNorm(x) ) )
        x = x + self.drop_path(self.attn(self.norm1(x), h, w))

        # --------------- FFN branch ---------------
        # x = x + DropPath( FFN( LayerNorm(x) ) )
        x = x + self.drop_path(self.ffn(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        """Compact parameter summary for ``print(model)``."""
        return (
            f"embed_dim={self.norm1.normalized_shape[0]}, num_heads={self.attn.num_heads}, "
            f"drop_path={self.drop_path.drop_prob if isinstance(self.drop_path, DropPath) else 0.0:.4f}"
        )
