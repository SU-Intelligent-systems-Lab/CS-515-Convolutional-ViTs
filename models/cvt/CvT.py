"""
CvT — Convolutional Vision Transformer (top-level model) (See: https://arxiv.org/abs/2103.15808).

This module is the root `nn.Module` that wires together the full CvT architecture as described in the paper.
It owns:

1.  Three `CvTStage` modules: each performs spatial downsampling via `ConvTokenEmbedding` followed by a stack 
    of `CvTBlock` modules.

2.  A global stochastic-depth schedule: a single linearly-spaced list of drop probabilities spanning all `sum(depths)` 
    blocks, sliced and distributed to each stage at construction time.

3.  A learnable CLS token: a single `embed_dim`-dimensional vector prepended to the token sequence *after* Stage 3 
    (not at the input like ViT). This placement is a deliberate design choice in CvT: by the time the CLS token is 
    introduced the spatial tokens already carry rich, multi-scale context, so the CLS token can aggregate it 
    efficiently with just the 10 Stage-3 blocks.

4.  A LayerNorm + Linear MLP head: projects the CLS token from `embed_dims[-1]` to `num_classes` logits.


Architecture summary
---------------------
    Input (B, 3, H, W)
      │
      │- Stage 1:   ConvEmbed(k=7, s=4) + 1 x CvTBlock -> (B, 256, 64) + (16,16)
      │             reshape -> (B, 64, 16, 16)
      │- Stage 2:   ConvEmbed(k=3, s=2) + 2 x CvTBlock -> (B, 64, 192) + (8,8)
      │             reshape -> (B, 192, 8, 8)
      │- Stage 3:   ConvEmbed(k=3, s=2) + 10 x CvTBlock -> (B, 16, 384) + (4,4)
           │
           │- prepend CLS token -> (B, 17, 384)
           │- LayerNorm(384)
           │- extract CLS token -> (B, 384)
           │- Linear(384, 200) -> (B, 200)   # (logits)

Weight initialization
----------------------

- `Conv2d`: truncated normal (sigma=0.02) or Kaiming normal
- `Linear`: truncated normal (sigma=0.02)
- `LayerNorm`: weight=1, bias=0
- `CLS` token: zeros

"""
from functools import partial
import torch
import torch.nn as nn
from torch import Tensor
from models.cvt.stage import CvTStage
from parameters import ModelConfig


class CvT(nn.Module):
    """
    Convolutional Vision Transformer.

    A hierarchical vision transformer with three stages, each performing convolutional token embedding followed by a
    stack of CvT Transformer Blocks. A learnable CLS token is injected after Stage 3 and used for classification.
    Shape:
        - Input: (B, in_channels, H, W)
        - Output: (B, num_classes) - raw (unnormalized) logits
    Args:
        cfg:    A `ModelConfig` dataclass instance that carries all architecture hyperparameters. Using a dataclass
                rather than individual keyword arguments keeps the constructor signature stable as the config evolves.

    Example:
        >>> cfg = ModelConfig()
        >>> model = CvT(cfg)
        >>> x = torch.randn(2, 3, 64, 64)
        >>> logits = model(x)
        >>> logits.shape
        torch.Size([2, 200])
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()

        self._cfg = cfg
        total_depth: int = sum(cfg.depths)

        # ----------- Global stochastic-depth schedule -----------
        # Linearly spaced from 0 to drop_path_rate across all blocks. Each stage receives the appropriate slice.
        dpr: list[float] = [float(x) for x in torch.linspace(0, cfg.drop_path_rate, total_depth)]

        # ------------------------ Stage 1 -----------------------
        # Operates on the raw image, large kernel for aggressive downsampling.
        self.stage1 = CvTStage(
            in_channels=cfg.in_channels,
            embed_dim=cfg.embed_dims[0],
            depth=cfg.depths[0],
            num_heads=cfg.num_heads[0],
            kernel_size_embed=cfg.kernel_size_embed,
            stride_embed=cfg.stride_embed,
            padding_embed=cfg.padding_embed,
            kernel_size_proj=cfg.kernel_size_proj,
            stride_kv=cfg.stride_kv,
            mlp_ratio=cfg.mlp_ratio,
            drop_rate=cfg.drop_rate,
            attn_drop_rate=cfg.attn_drop_rate,
            drop_path_probs=dpr[: cfg.depths[0]],
            qkv_bias=cfg.qkv_bias,
        )

        # ------------------------ Stage 2 -----------------------
        # Here: Input channels = embed_dim of Stage 1.
        self.stage2 = CvTStage(
            in_channels=cfg.embed_dims[0],
            embed_dim=cfg.embed_dims[1],
            depth=cfg.depths[1],
            num_heads=cfg.num_heads[1],
            kernel_size_embed=3,
            stride_embed=2,
            padding_embed=1,
            kernel_size_proj=cfg.kernel_size_proj,
            stride_kv=cfg.stride_kv,
            mlp_ratio=cfg.mlp_ratio,
            drop_rate=cfg.drop_rate,
            attn_drop_rate=cfg.attn_drop_rate,
            drop_path_probs=dpr[cfg.depths[0]: cfg.depths[0] + cfg.depths[1]],
            qkv_bias=cfg.qkv_bias,
        )

        # ------------------------ Stage 3 -----------------------
        # Here: Input channels = embed_dim of Stage 2.
        self.stage3 = CvTStage(
            in_channels=cfg.embed_dims[1],
            embed_dim=cfg.embed_dims[2],
            depth=cfg.depths[2],
            num_heads=cfg.num_heads[2],
            kernel_size_embed=3,
            stride_embed=2,
            padding_embed=1,
            kernel_size_proj=cfg.kernel_size_proj,
            stride_kv=cfg.stride_kv,
            mlp_ratio=cfg.mlp_ratio,
            drop_rate=cfg.drop_rate,
            attn_drop_rate=cfg.attn_drop_rate,
            drop_path_probs=dpr[cfg.depths[0] + cfg.depths[1]:],
            qkv_bias=cfg.qkv_bias,
        )

        # ----------------------- CLS token ----------------------
        # A single learnable vector prepended to the Stage-3 token sequence.
        # Shape: (1, 1, embed_dims[-1]) - broadcast over the batch dimension.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dims[2]))

        # ------------------ Classification head -----------------
        # Applied to the CLS token after Layer Normalisation.
        self.norm = nn.LayerNorm(cfg.embed_dims[2])
        self.head = nn.Linear(cfg.embed_dims[2], cfg.num_classes)

        # ----------------- Weight Initialization ----------------
        self._init_weights(cfg.init_weights)

    def forward(self, x: Tensor) -> Tensor:
        """
        Classify a batch of images.

        Args:
            x: Image batch of shape (B, in_channels, H, W).

        Returns:
            Logit tensor of shape (B, num_classes).
        """
        B = x.shape[0]

        # ------------------------ Stage 1 -----------------------
        # (B, 3, H, W) -> tokens (B, N1, D1), spatial (h1, w1)
        tokens, (h, w) = self.stage1(x)

        # Reshape tokens back to a 2-D feature map for Stage 2's ConvEmbed.
        # (B, N1, D1) -> (B, D1, h1, w1)
        x = tokens.transpose(1, 2).reshape(B, self._cfg.embed_dims[0], h, w)

        # ------------------------ Stage 2 -----------------------
        # (B, D1, h1, w1) -> tokens (B, N2, D2), spatial (h2, w2)
        tokens, (h, w) = self.stage2(x)

        # Reshape for Stage 3's ConvEmbed.
        # (B, N2, D2) -> (B, D2, h2, w2)
        x = tokens.transpose(1, 2).reshape(B, self._cfg.embed_dims[1], h, w)

        # ------------------------ Stage 3 -----------------------
        # (B, D2, h2, w2) -> tokens (B, N3, D3), spatial (h3, w3)
        tokens, _ = self.stage3(x)

        # ----------------- CLS token injection ------------------
        # Expand the (1, 1, D3) parameter to (B, 1, D3) and prepend.
        cls = self.cls_token.expand(B, -1, -1)   # (B, 1, D3)
        tokens = torch.cat([cls, tokens], dim=1)   # (B, 1 + N3, D3)

        # --------------- LayerNorm + extract CLS ----------------
        tokens = self.norm(tokens)
        cls_out = tokens[:, 0]                  # (B, D3)

        # ----------------- Classification head ------------------
        logits = self.head(cls_out)             # (B, num_classes)
        return logits

    def _init_weights(self, scheme: str) -> None:
        """
        Initialize all submodule weights according to `scheme`.

        Args:
            scheme: "trunc_normal" (default, ViT-style) or "kaiming" (ResNet-style).
        """

        def _trunc_normal(tensor: Tensor, std: float = 0.02) -> Tensor:
            """
            Fill `tensor` with values from a truncated normal distribution.

            Values are drawn from N(0, std^2) and clipped to -+2sigma. This is the standard initialization for ViT models.

            Args:
                tensor: The tensor to initialize in-place.
                std: Standard deviation of the underlying normal distribution.

            Returns:
                The initialized tensor (same object, modified in-place).
            """
            with torch.no_grad():
                # Draw from normal, then clamp to -+2sigma.
                nn.init.normal_(tensor, mean=0.0, std=std)
                tensor.clamp_(-2 * std, 2 * std)
            return tensor

        trunc_normal_ = partial(_trunc_normal, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                if scheme == "trunc_normal":
                    trunc_normal_(module.weight)
                else:
                    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Conv2d):
                if scheme == "trunc_normal":
                    trunc_normal_(module.weight)
                else:
                    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # CLS token: start at zero; the model will learn to populate it.
        nn.init.zeros_(self.cls_token)

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extra_repr(self) -> str:
        """Compact summary shown in ``print(model)``."""
        cfg = self._cfg
        return (
            f"embed_dims={cfg.embed_dims}, depths={cfg.depths}, num_heads={cfg.num_heads}, "
            f"num_classes={cfg.num_classes}, params={self.num_parameters:,}"
        )
