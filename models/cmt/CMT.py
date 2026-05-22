"""
CMT: Convolutional Neural Networks Meet Vision Transformers (top-level model) (See: https://arxiv.org/abs/2107.06263).

This module is the root `nn.Module` that wires the full CMT architecture:
    1.  CMTStem: three 3x3 convs, stride-2 on the first only.
    2.  Four CMTStage: each contains PatchEmbedding + a stack of CMTBlocks (LPU + SR-MHSA + IRFFN).
    3.  A `GlobalAvgPool + Linear` classification head: Unlike CvT (which uses a learnable CLS token), CMT uses
        global average pooling over all output tokens, which is more parameter-efficient and better suited for its
        4-stage hierarchy.

Architecture summary (CMT-Ti, Tiny ImageNet 64x64)
----------------------------------------------------
    Input           (B,   3,  64,  64)
    CMTStem         (B,  16,  32,  32)   stride-2 first conv, 3 conv layers total
    Stage 1
          PatchEmbed    (B, 256,  46)        H=16, W=16   (k=2 s=2, 32->16)
          2 x CMTBlock  (B, 256,  46)
          reshape       (B,  46,  16,  16)
    Stage 2
          PatchEmbed    (B,  64,  92)        H=8,  W=8
          2 x CMTBlock  (B,  64,  92)
          reshape       (B,  92,   8,   8)
    Stage 3
          PatchEmbed    (B,  16, 184)        H=4,  W=4
          10 x CMTBlock (B,  16, 184)
          reshape       (B, 184,   4,   4)
    Stage 4
          PatchEmbed    (B,   4, 368)        H=2,  W=2
          2 x CMTBlock  (B,   4, 368)
    GlobalAvgPool   (B, 368)        # mean over 4 tokens
    LayerNorm       (B, 368)
    Linear          (B, 200)        # logits


Fixed per-stage hyperparameters (all CMT variants)
------------------------------------------------------------
    num_heads : (1, 2, 4, 8)   — not in ModelConfig; hardcoded here
    sr_ratios : (8, 4, 2, 1)   — not in ModelConfig; hardcoded here

These values do not change across Ti / XS / S / B variants and therefore are not exposed as CLI arguments.


Variant-specific hyperparameters (from `ModelConfig`)
--------------------------------------------------------------------------------
    cmt_stem_channels  : int
    cmt_channel_dims   : tuple[int, ...]   length 4
    cmt_depths         : tuple[int, ...]   length 4
    cmt_mlp_ratios     : tuple[float, ...] length 4


Shared hyperparameters (also used by CvT)
--------------------------------------------------------------------
    qkv_bias        : bool
    drop_rate       : float
    attn_drop_rate  : float
    drop_path_rate  : float
    init_weights    : str


CMT vs CvT — key differences
------------------------------
| Aspect             | CvT                         | CMT                          |
|--------------------|-----------------------------|------------------------------|
| Stages             | 3                           | 4                            |
| Stem               | large-kernel ConvEmbed      | 3x 3x3 Conv (BN+GELU)        |
| Token embedding    | overlapping strided conv    | non-overlapping Patch Embed  |
| Attention          | ConvQ/K/V (all conv proj)   | LMHSA (full Q, reduced KV)   |
| FFN                | standard two-linear MLP     | IRFFN (DWConv in hidden dim) |
| Position encoding  | implicit via ConvEmbed      | implicit via LPU DWConv (Not Implemented)|
| Classification     | learnable CLS token         | GlobalAvgPool                |
| Positional info    | stride in all projections   | LPU before every block       |
"""
from functools import partial
import torch
import torch.nn as nn
from torch import Tensor
from models.cmt.cmt_stem import CMTStem
from models.cmt.cmt_stage import CMTStage
from parameters import Config


# Fixed per-stage values, identical across all CMT variants.
_NUM_HEADS: tuple[int, ...] = (1, 2, 4, 8)
_SR_RATIOS:  tuple[int, ...] = (8, 4, 2, 1)


class CMT(nn.Module):
    """
    Convolutional Neural Networks Meet Vision Transformers.

    A four-stage hierarchical hybrid model. Constructed from a `Config` that includes `ModelConfig`
    (architecture hyperparameters) and a `DataConfig` (dataset metadata such as `num_classes` and `in_channels`).

    Args:
        cfg: `Config` instance, includes `ModelConfig` and `DataConfig`.

    Shape:
        - Input:  (B, in_channels, H, W)
        - Output: (B, num_classes) raw unnormalized logits

    Example:
        >>> model = CMT(cfg)
        >>> x = torch.randn(2, 3, 64, 64)
        >>> logits = model(x)
        >>> logits.shape
        torch.Size([2, 200])
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self._cfg = cfg

        # Unpack variant-specific fields (resolved by ModelConfig)
        stem_channels = cfg.model.cmt_stem_channels  # e.g. 16 for Ti
        channels_dims = cfg.model.cmt_channel_dims  # (46, 92, 184, 368) for Ti
        depths = cfg.model.cmt_depths  # (2, 2, 10, 2) for Ti
        mlp_ratios = cfg.model.cmt_mlp_ratios  # (3.6, 3.6, 3.6, 3.6) for Ti

        num_classes = cfg.data.num_classes
        in_channels = cfg.data.in_channels

        total_depth: int = sum(depths)

        # Global stochastic-depth (Drop Path) schedule: One linearly-spaced probability per block across the entire
        # model. => dpr[0] = 0.0 (first block is never dropped)
        #           dpr[-1] = drop_path_rate (last block has the highest probability)
        dpr: list[float] = [float(x) for x in torch.linspace(0, cfg.model.drop_path_rate, total_depth)]

        # CMT Stem: Three 3x3 convs (GELU->BN), stride-2 on the first only. Output: (B, stem_channel, H/2, W/2)
        self.stem = CMTStem(in_channels=in_channels, stem_channels=stem_channels)

        # Four CMT Stages
        # Per-stage in_dim values:
        #   Stage 1 -> stem_channels (stem output, no channel doubling)
        #   Stage k -> channels_dims[k-2]  (previous stage's embedding dim)
        # PatchEmbedding uses kernel=stride=2 for every stage.
        # num_heads and sr_ratios are fixed constants, not config fields.

        cursor = 0  # current position in the global dpr list

        self.stage1 = CMTStage(
            in_dim=stem_channels,
            embed_dim=channels_dims[0],
            depth=depths[0],
            num_heads=_NUM_HEADS[0],
            sr_ratio=_SR_RATIOS[0],
            mlp_ratio=mlp_ratios[0],
            qkv_bias=cfg.model.qkv_bias,
            drop_rate=cfg.model.drop_rate,
            attn_drop_rate=cfg.model.attn_drop_rate,
            drop_path_probs=dpr[cursor: cursor + depths[0]],
        )
        cursor += depths[0]

        self.stage2 = CMTStage(
            in_dim=channels_dims[0],
            embed_dim=channels_dims[1],
            depth=depths[1],
            num_heads=_NUM_HEADS[1],
            sr_ratio=_SR_RATIOS[1],
            mlp_ratio=mlp_ratios[1],
            qkv_bias=cfg.model.qkv_bias,
            drop_rate=cfg.model.drop_rate,
            attn_drop_rate=cfg.model.attn_drop_rate,
            drop_path_probs=dpr[cursor: cursor + depths[1]],
        )
        cursor += depths[1]

        self.stage3 = CMTStage(
            in_dim=channels_dims[1],
            embed_dim=channels_dims[2],
            depth=depths[2],
            num_heads=_NUM_HEADS[2],
            sr_ratio=_SR_RATIOS[2],
            mlp_ratio=mlp_ratios[2],
            qkv_bias=cfg.model.qkv_bias,
            drop_rate=cfg.model.drop_rate,
            attn_drop_rate=cfg.model.attn_drop_rate,
            drop_path_probs=dpr[cursor: cursor + depths[2]],
        )
        cursor += depths[2]

        self.stage4 = CMTStage(
            in_dim=channels_dims[2],
            embed_dim=channels_dims[3],
            depth=depths[3],
            num_heads=_NUM_HEADS[3],
            sr_ratio=_SR_RATIOS[3],
            mlp_ratio=mlp_ratios[3],
            qkv_bias=cfg.model.qkv_bias,
            drop_rate=cfg.model.drop_rate,
            attn_drop_rate=cfg.model.attn_drop_rate,
            drop_path_probs=dpr[cursor: cursor + depths[3]],
        )

        # Classification head: Applied after GlobalAvgPool over the Stage-4 token sequence.
        self.norm = nn.LayerNorm(channels_dims[3])
        self.head = nn.Linear(channels_dims[3], num_classes)

        # Weight initialization
        self._init_weights(cfg.model.init_weights)

    def forward(self, x: Tensor) -> Tensor:
        """
        Classify a batch of images.

        Args:
            x: Image batch of shape (B, in_channels, H, W).

        Returns:
            Logit tensor of shape (B, num_classes).
        """
        B = x.shape[0]
        channels_dims = self._cfg.model.cmt_channel_dims

        #  Stem: (B, 3, H, W) -> (B, stem_channels, H/2, W/2)
        x = self.stem(x)

        # Stage 1: Feature map -> tokens, then reshape back for Stage 2's PatchEmbedding
        tokens, (h, w) = self.stage1(x)             # (B, N1, channels_dims[0])
        x = tokens.transpose(1, 2).reshape(B, channels_dims[0], h, w)

        # Stage 2
        tokens, (h, w) = self.stage2(x)
        x = tokens.transpose(1, 2).reshape(B, channels_dims[1], h, w)

        # Stage 3
        tokens, (h, w) = self.stage3(x)
        x = tokens.transpose(1, 2).reshape(B, channels_dims[2], h, w)

        # Stage 4: No reshape needed after Stage 4 — output goes straight to GlobalAvgPool.
        tokens, _ = self.stage4(x)          # (B, N4, channels_dims[3])

        # Global Average Pooling: Mean over all spatial tokens -> (B, channels_dims[3])
        x = tokens.mean(dim=1)

        # LayerNorm + linear head
        x = self.norm(x)                    # (B, channels_dims[3])
        logits = self.head(x)               # (B, num_classes)

        return logits

    def _init_weights(self, scheme: str) -> None:
        """
        Initialize all submodule weights.

        Args:
            scheme: "trunc_normal": truncated normal (ViT-style default). "kaiming": Kaiming normal (He initialization).
        """
        def _trunc_normal(tensor: Tensor, std: float = 0.02) -> Tensor:
            """
            Fill `tensor` with values from a truncated normal distribution.

            Values are drawn from N(0, std^2) and clipped to -+2sigma. This is the standard initialization for ViTs.

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

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extra_repr(self) -> str:
        """Compact summary shown in `print(model)`."""
        cfg = self._cfg
        return (
            f"variant={cfg.model.cmt_variant}, channel_dims={cfg.model.cmt_channel_dims}, "
            f"depths={cfg.model.cmt_depths}, num_heads={_NUM_HEADS}, sr_ratios={_SR_RATIOS}, "
            f"num_classes={cfg.data.num_classes}, params={self.num_parameters:,}"
        )
