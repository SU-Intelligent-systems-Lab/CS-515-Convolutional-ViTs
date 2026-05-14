"""
Convolutional Multi-Head Self-Attention for CvT.

This module is the core novelty of CvT. In a standard Vision Transformer (ViT), Q, K, and V are
produced by three linear layers applied to the flat token sequence. CvT replaces those linear layers with
depthwise separable convolutions that operate on the 2-D spatial layout of the tokens.

Why convolutions for Q / K / V?
--------------------------------
- Local inductive bias — the conv captures neighborhood context before attention aggregates globally. A token's
  query or key already knows what its surrounding tokens look like.
- Fewer parameters, more efficiency — depthwise separable convolutions are far cheaper than dense linear projections
  for the same receptive field.
- Implicit positional encoding — because the convolution sees spatial position through its local receptive field,
  no explicit position embeddings are required (unlike ViT).
- Key/Value downsampling — K and V use `stride = stride_kv > 1`, which reduces the sequence length before the 
  expensive O(N^2) attention, acting as a cheap spatial pooling.

Depthwise Separable Convolution Projection
------------------------------------------
For each of Q, K, V the projection is:

(B, N, C) -> reshape -> (B, C, H, W)
            ↓
      Depthwise Conv2d            <- one filter per channel (groups=C)
      BatchNorm2d                 <- stabilize across spatial positions
      GELU                        <- non-linearity
      Pointwise Conv2d (1x1)      <- mix channels
            ↓
(B, C', H', W') -> reshape -> (B, H'*W', C')

For Q the stride is always 1 (N' = N).
For K and V the stride is `stride_kv` which is bigger than 1 (N' = H'*W' ≤ N) from the "Squeezed convolutional
projection" stated in the paper. The smaller K/V sequence is what makes CvT attention sub-quadratic without any
approximation.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from models.cvt.conv_projection import ConvProjection


class ConvAttention(nn.Module):
    """
    Convolutional Multi-Head Self-Attention (the CvT core operation).

    Replaces the three linear Q/K/V projections of standard self-attention with `ConvProjection` modules
    (depthwise separable convolutions). After projection, standard scaled dot-product attention is computed between
    Q (full sequence) and the downsampled K, V.

    Attention formula: Attention(Q, K, V) = softmax( Q x K_T / sqr_root(d_head) ) x V

    where `d_head = embed_dim // num_heads`.

    Args:
        embed_dim: Total embedding dimension C. Must be divisible by `num_heads`.
        num_heads: Number of parallel attention heads.
        kernel_size: Kernel size for all three `ConvProjection` modules.
        stride_kv: Stride used for K and V projections. A value of 2 reduces the K/V sequence length by ~4x
                   (each spatial dim / 2), making attention roughly 4x cheaper. Q always uses stride 1.
        attn_drop: Dropout probability applied to the attention weight matrix after softmax. Defaults to 0.0
        proj_drop: Dropout probability applied to the final output projection. Defaults to 0.0
        qkv_bias: If `True`, the pointwise conv in each `ConvProjection` has a learnable bias.

    Shape:
        - Input:  `(B, N, C)` tokens + `(H, W)` spatial shape
        - Output: `(B, N, C)` attended tokens (sequence length unchanged)

    Raises:
        AssertionError: If `embed_dim` is not divisible by `num_heads`.

    Example:
        >>> attn = ConvAttention(embed_dim=64, num_heads=1, kernel_size=3, stride_kv=2)
        >>> x = torch.randn(2, 256, 64)   # Stage-1 tokens
        >>> out = attn(x, h=16, w=16)
        >>> out.shape
        torch.Size([2, 256, 64])
    """

    def __init__(self, embed_dim: int, num_heads: int, kernel_size: int = 3, stride_kv: int = 2,
                 attn_drop: float = 0.0, proj_drop: float = 0.0, qkv_bias: bool = True) -> None:
        super().__init__()

        assert (embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)  # pre-compute 1/sqr_root(d_head) denominator

        pad = kernel_size // 2   # same-spatial padding for Q (stride=1)

        # Q: stride = 1, so output sequence length stays N.
        self.proj_q = ConvProjection(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=pad,
            bias=qkv_bias,
        )

        # K, V: stride = stride_kv, so output sequence length shrinks.
        self.proj_k = ConvProjection(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=kernel_size,
            stride=stride_kv,
            padding=pad,
            bias=qkv_bias,
        )
        self.proj_v = ConvProjection(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=kernel_size,
            stride=stride_kv,
            padding=pad,
            bias=qkv_bias,
        )

        self.attn_drop = nn.Dropout(attn_drop)

        # Final linear projection to mix head outputs.
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, h: int, w: int) -> Tensor:
        """Compute convolutional multi-head self-attention.

        Steps:
            1. Project tokens into Q (stride=1), K and V (stride=stride_kv)
               via depthwise separable convolutions on the 2-D feature map.
            2. Split each of Q, K, V into `num_heads` heads.
            3. Compute scaled dot-product attention scores.
            4. Apply softmax + dropout -> weighted sum over V.
            5. Concatenate heads -> linear output projection + dropout.

        Args:
            x: Input token sequence `(B, N, C)` where `N = H * W`.
            h: Spatial height of the token map (needed to reshape for conv).
            w: Spatial width of the token map.

        Returns:
            Attended token sequence of shape `(B, N, C)`.
        """
        B, N, C = x.shape

        # -------- 1. Convolutional Q / K / V projections --------
        # Q: (B, N,  C)   — full spatial resolution
        # K: (B, N', C)   — spatially downsampled (N' ≤ N)
        # V: (B, N', C)   — spatially downsampled (N' ≤ N)
        q, _, _ = self.proj_q(x, h, w)
        k, h_kv, w_kv = self.proj_k(x, h, w)
        v, _, _ = self.proj_v(x, h, w)

        N_kv = h_kv * w_kv   # reduced sequence length for K / V

        # ----------------- 2. Split into heads ------------------
        # Reshape to (B, num_heads, seq_len, head_dim) for batched matmul.
        # This is important to respect the spatial size of the tokens since q and k,v have different spatial sizes.
        def _split_heads(t: Tensor, seq_len: int) -> Tensor:
            return t.reshape(B, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)       # (B, H, seq, d_head)

        q = _split_heads(q, N)
        k = _split_heads(k, N_kv)
        v = _split_heads(v, N_kv)

        # -------- 3. Scaled dot-product attention scores --------
        # softmax( Q x K_T / sqr_root(d_head) )

        # Dot product of query and key but respecting the different shapes by transposing key => q x k_T
        # => (B, num_heads, N, d_head) x (B, num_heads, d_head, N_kv) => (B, num_heads, N, N_kv)
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # ---------------- 4. Weighted sum over V ----------------
        # Attention(Q, K, V) = softmax( Q x K_T / sqr_root(d_head) ) x V
        # Attention(Q, K, V) = (Output of 3.) x V

        # (B, num_heads, N, N_kv) x (B, num_heads, N_kv, d_head) => (B, num_heads, N, d_head)
        out = torch.matmul(attn, v)

        # ------- 5. Concatenate heads + output projection -------
        # (B, num_heads, N, d_head) -> (B, N, num_heads, d_head) -> (B, N, C)
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)
        out = self.proj_drop(self.out_proj(out))

        return out

    def extra_repr(self) -> str:
        """Compact parameter summary for `print(model)`."""
        return (
            f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, head_dim={self.head_dim}, "
            f"stride_kv={self.proj_k.stride}"
        )
    