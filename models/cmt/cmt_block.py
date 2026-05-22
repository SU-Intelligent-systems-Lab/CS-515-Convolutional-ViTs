"""
CMT Block — the repeating unit of every CMT stage.

Position in the CMT architecture
----------------------------------
A `CMTBlock` is the fundamental transformer unit stacked `depth` times inside each of the four CMT stages. It wires 
together the three submodules implemented in the previous steps, adding pre-LayerNorm, residual connections, and 
stochastic depth regularization.

Key design notes
-----------------
1.  LPU has no surrounding norm or residual in this block. The LPU already adds its own internal residual
    (`x = x + DWConv(x)`). It is not wrapped in a pre-norm because it is purely a local feature injector — applying
    LN before it would normalize away the very spatial statistics the DWConv is trying to encode.

2.  Pre-norm pattern for MHSA and FFN. LayerNorm is applied to `x` before each submodule, not after. This leads to
    more stable training gradients compared to the original post-norm Transformer: the residual branch always carries
    the raw (un-normalized) signal, so the gradient can flow cleanly through the skip connection regardless of how
    the submodule's activations are scaled.

3.  DropPath (Stochastic Depth) is applied to the output of both MHSA and IRFFN before adding the residual. Each block
    in the network receives a different drop probability from the global linear schedule computed in `CMTStage`:
    earlier blocks get lower probability, later blocks get higher. This regularizes deep models more aggressively
    where it costs the least (deep blocks have many redundant paths).

4.  Spatial shape (H, W) threading. Every submodule that contains a depthwise convolution (LPU, spatial reduction in
    LMHSA, IRFFN's DWConv) needs to know the 2-D layout of the token sequence to perform the reshape.
    `CMTBlock.forward` receives `H` and `W` and threads them through to all three submodules.

Reuse from CvT
--------------
`DropPath` is imported directly from `models.cvt.drop_path`, no duplication. This is the only CvT component shared 
with CMT; everything else (attention, FFN, embedding) differs enough to warrant independent implementations.
"""
import torch.nn as nn
from torch import Tensor
from models.cvt.drop_path import DropPath
from models.cmt.cmt_lpu import CMT_LPU
from models.cmt.cmt_lmhsa import CMT_LMHSA
from models.cmt.cmt_irffn import CMT_IRFFN


class CMTBlock(nn.Module):
    """
    One CMT Transformer Block.

    Composes three submodules in sequence (LPU, LMHSA, IRFFN) with pre-LayerNorm and DropPath-regularized residual 
    connections around the attention and FFN branches.

    Args:
        dim: Token embedding dimension C for this stage.
        num_heads: Number of attention heads. Must evenly divide `dim`.
        sr_ratio: Spatial reduction factor for Keys and Values in LMHSA. `sr_ratio=1` means no reduction (standard 
                  MHSA). Typical CMT-Ti values: 8, 4, 2, 1 for stages 1–4.
        mlp_ratio: Hidden-dimension expansion factor for the IRFFN: `hidden = int(dim × mlp_ratio)`.
        qkv_bias: Learnable bias in the Q, KV, and output projections of LMHSA.
        drop: Dropout probability for IRFFN output and LMHSA output projections.
        attn_drop: Dropout probability on the attention weight matrix inside LMHSA.
        drop_path: Stochastic-depth drop probability for this specific block. Supplied by `CMTStage` as a slice of the 
                   global linear schedule.

    Shape:
        - Input:  (B, N, C) + (H, W)  where N = H × W
        - Output: (B, N, C)

    Example:
        >>> block = CMTBlock(dim=46, num_heads=1, sr_ratio=8, mlp_ratio=3.6)
        >>> x = torch.randn(2, 64, 46)   # stage 1: 8x8 spatial, 46-dim tokens
        >>> out = block(x, H=8, W=8)
        >>> out.shape
        torch.Size([2, 64, 46])
    """

    def __init__(self, dim: int, num_heads: int, sr_ratio: int = 1, mlp_ratio: float = 4.0, qkv_bias: bool = True, 
                 drop: float = 0.0, attn_drop: float = 0.0, drop_path: float = 0.0) -> None:
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.sr_ratio = sr_ratio
        self.mlp_ratio = mlp_ratio

        # LPU, no surrounding norm, has its own internal residual
        self.lpu = CMT_LPU(dim=dim)

        # Pre-norm + LMHSA
        self.norm1 = nn.LayerNorm(dim)
        self.attn = CMT_LMHSA(dim=dim, num_heads=num_heads, sr_ratio=sr_ratio, qkv_bias=qkv_bias, 
                              attn_drop=attn_drop, proj_drop=drop)

        # Pre-norm + IRFFN
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = CMT_IRFFN(dim=dim, mlp_ratio=mlp_ratio, drop=drop)

        # Stochastic depth, shared for both residual branches. Identity when drop_path=0.0 (no overhead at inference).
        # Reuses CvT's DropPath: same Stochastic Depth implementation.
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        """
        Apply one CMT Block to a token sequence.

        Args:
            x: Token sequence of shape (B, N, C) where N = H × W.
            H: Spatial height of the current feature map (passed to all DWConv ops).
            W: Spatial width of the current feature map.

        Returns:
            Transformed token sequence of shape (B, N, C).
        """
        # 1- LPU - Local Perception Unit: injects local spatial context via a residual depthwise 3×3 conv.
        # No surrounding norm, LPU manages its own residual.
        x = self.lpu(x, H, W)

        # 2- Pre-norm -> LMHSA -> DropPath -> residual
        # LayerNorm is applied to x before attention. The original x (before norm) is used as the residual,
        # preserving unscaled gradient flow.
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))

        # 3- Pre-norm -> IRFFN -> DropPath -> residual
        # Same pre-norm pattern. IRFFN returns its output without a residual; the block applies it here after DropPath.
        x = x + self.drop_path(self.ffn(self.norm2(x), H, W))

        return x

    @property
    def drop_prob(self) -> float:
        """Stochastic depth drop probability (0.0 if DropPath is disabled)."""
        return self.drop_path.drop_prob if isinstance(self.drop_path, DropPath) else 0.0

    def extra_repr(self) -> str:
        """Compact summary shown in `print(model)`."""
        return (
            f"dim={self.dim}, num_heads={self.num_heads}, "
            f"sr_ratio={self.sr_ratio}, mlp_ratio={self.mlp_ratio}, "
            f"drop_path={self.drop_prob:.4f}"
        )
