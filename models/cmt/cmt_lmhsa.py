"""
CMT Lightweight Multi-Head Self-Attention (LMHSA).

Position in the CMT architecture
----------------------------------
LMHSA is the second submodule inside every CMT Block, running after the LPU and before the IRFFN. The parent CMTBlock
applies pre-LayerNorm before calling this module and adds the residual after: x = x + DropPath(LMHSA(LN(x), H, W))

From the CMT paper (Section 3.2), the key idea is to reduce the sequence length of Keys and Values by a factor of
`k_i` (the spatial reduction ratio for stage i) while keeping Queries at full resolution:
    Q = X · W_Q                (B, N,   d) - full resolution
    K = SR(X, k_i) · W_K       (B, N_r, d) - spatially reduced
    V = SR(X, k_i) · W_V       (B, N_r, d) - spatially reduced
where the Spatial Reduction (SR) is a depthwise convolution with (kernel_size = stride = `k_i`) which maps the
spatial grid from (H, W) to (H/k_i, W/k_i), reducing the token count from N = HxW to N_r = (H/k_i)(W/k_i) = N/k_i^2.

The attention matrix is then (B, heads, N, N_r) instead of (B, heads, N, N), reducing the attention cost from O(N^2)
to O(N · N/k_i^2).

Reduction rates k_i per stage (fixed across all CMT variants, Table 1):
    Stage 1: k=8 -> N_r = N/64
    Stage 2: k=4 -> N_r = N/16
    Stage 3: k=2 -> N_r = N/4
    Stage 4: k=1 -> no reduction (standard MHSA)

Number of heads H_i per stage (fixed across all CMT variants, Table 1):
    Stage 1: H=1
    Stage 2: H=2
    Stage 3: H=4
    Stage 4: H=8

Note on relative position embedding
-------------------------------------
The CMT paper mentions incorporating a relative position embedding into LMHSA. This is NOT implemented here. It is
left as a future extension. The current implementation is the core attention mechanism without positional bias.
"""
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CMT_LMHSA(nn.Module):
    """
    Lightweight Multi-Head Self-Attention for CMT.

    Queries attend at full resolution while Keys and Values are computed on a spatially compressed representation,
    reducing the attention cost from O(N^2) to O(N · N_r) where N_r = N / k^2 and k is the spatial reduction ratio.

    Args:
        dim: Token embedding dimension C. Must be divisible by `num_heads`.
        num_heads: Number of parallel attention heads `H`. Fixed per stage: (1, 2, 4, 8) for stages 1–4.
        sr_ratio: Spatial reduction factor `k` for Keys and Values. Fixed per stage: (8, 4, 2, 1) for stages 1–4.
                  `k=1` means no reduction (stage 4).
        qkv_bias: If `True`, add learnable bias to Q, KV, and output projections.
        attn_drop: Dropout probability on the attention weight matrix.
        proj_drop: Dropout probability on the output projection.

    Shape:
        - Input:  (B, N, C) + (H, W)  where N = H x W
        - Output: (B, N, C)

    Example:
        >>> import torch
        >>> attn = CMT_LMHSA(dim=46, num_heads=1, sr_ratio=8)
        >>> x = torch.randn(2, 256, 46)    # stage 1: 16x16 tokens
        >>> out = attn(x, H=16, W=16)
        >>> out.shape
        torch.Size([2, 256, 46])
    """

    def __init__(self, dim: int, num_heads: int, sr_ratio: int = 1, qkv_bias: bool = True, attn_drop: float = 0.0,
                 proj_drop: float = 0.0) -> None:
        super().__init__()

        assert dim % num_heads == 0, f"dim={dim} must be divisible by num_heads={num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.sr_ratio = sr_ratio
        self.scale = self.head_dim ** -0.5    # 1 / sqrt(d_head)

        # 1. Query projection (full resolution):  Maps (B, N, C) -> (B, N, C). The result is reshaped to multi-head
        # format (B, h, N, head_dim) inside forward().
        self.q = nn.Linear(dim, dim, bias=qkv_bias)

        # 2. Spatial Reduction for K and V
        # From the paper: "a depth-wise convolution with stride k_i to reduce the spatial resolution of K and V."
        # kernel_size = stride = sr_ratio -> non-overlapping reduction windows.
        # Only instantiated when sr_ratio > 1; when sr_ratio=1 (stage 4), K and V are computed directly from the
        # full sequence.
        if sr_ratio > 1:
            self.sr = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=sr_ratio, stride=sr_ratio, groups=dim)
            self.sr_norm = nn.LayerNorm(dim)

        # 3. Key + Value projection (reduced resolution): Single linear maps the reduced sequence to 2*dim, then split
        # into K, V. Input sequence length is N_r = N / r^2 (after spatial reduction).
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        # 4. Output projection: Recombines all heads back to (B, N, C).
        self.proj = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def _spatial_reduce(self, x: Tensor, H: int, W: int) -> Tensor:
        """
        Apply spatial reduction to obtain a compressed sequence for K and V.

        Steps:
            1. Reshape (B, N, C) -> (B, C, H, W)
            2. DWConv(kernel=k, stride=k) -> (B, C, H/k, W/k)
            3. Flatten + transpose -> (B, N_r, C)
            4. LayerNorm

        Args:
            x: Token sequence (B, N, C).
            H: Spatial height before reduction.
            W: Spatial width before reduction.

        Returns:
            Reduced token sequence (B, N_r, C)
            where N_r = (H / sr_ratio) * (W / sr_ratio).
        """
        B, _N, C = x.shape

        # (B, N, C) -> (B, C, H, W)
        x_2d = x.transpose(1, 2).reshape(B, C, H, W)

        # (B, C, H, W) -> (B, C, H/k, W/k)
        x_2d = self.sr(x_2d)

        # (B, C, H/k, W/k) -> (B, N_r, C)
        x_r = x_2d.flatten(2).transpose(1, 2)

        # LayerNorm over channel dimension
        return self.sr_norm(x_r)

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        """
        Compute Lightweight Multi-Head Self-Attention.

        Args:
            x: Token sequence (B, N, C) - already LayerNorm-ed by CMTBlock.
            H: Spatial height of the current feature map.
            W: Spatial width of the current feature map.

        Returns:
            Attended token sequence (B, N, C).
        """
        B, N, C = x.shape
        h = self.num_heads
        d = self.head_dim

        # 1. Query with full resolution
        # (B, N, C) -> (B, N, h, d) -> (B, h, N, d)
        q = self.q(x).reshape(B, N, h, d).permute(0, 2, 1, 3) 

        # 2. Spatial reduction of the input for K/V
        if self.sr_ratio > 1:
            x_r = self._spatial_reduce(x, H, W)    # (B, N_r, C)
        else:
            x_r = x                                 # (B, N, C) -> no reduction

        N_r = x_r.shape[1]

        # 3. Key and Value with reduced resolution
        # (B, N_r, C) -> (B, N_r, 2C) -> split -> each (B, h, N_r, d)
        kv = self.kv(x_r).reshape(B, N_r, 2, h, d).permute(2, 0, 3, 1, 4)

        k, v = kv.unbind(0)       # each (B, h, N_r, d)

        # 4. Scaled dot-product attention
        # attn (B, h, N, N_r): each of N queries attends to N_r key positions
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # (B, h, N, d) after weighting values
        out = attn @ v

        # 5. Merge heads and output projection
        # (B, h, N, d) -> (B, N, h, d) -> (B, N, h*d) = (B, N, C)
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        return out

    def extra_repr(self) -> str:
        """Compact summary shown in `print(model)`"""
        return f"dim={self.dim}, num_heads={self.num_heads}, head_dim={self.head_dim}, sr_ratio={self.sr_ratio}"
