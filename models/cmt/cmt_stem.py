"""
CMT Convolutional Stem.

Position in the CMT architecture
----------------------------------
The stem is the very first component that sees the raw input image. Before any transformer block, patch aggregation,
or attention mechanism, the stem performs an initial spatial downsampling and channel projection, handing a compact
feature map to Stage 1.

Full CMT data flow (Tiny ImageNet, 64x64, CMT-Ti):

    Raw image  (B, 3, 64, 64)
         │
    │---------------------------------------------------│
    │  CMT STEM  (this module)                          │
    │  Conv(3->32,  k=3, s=2, p=1) -> GELU -> BN        │
    │  Conv(32->32, k=3, s=1, p=1) -> GELU -> BN        │
    │  Conv(32->32, k=3, s=1, p=1) -> GELU -> BN        │
    │---------------------------------------------------│
         │  (B, 32, 32, 32)          <- stem output
         ↓
    Stage 1 PatchAggregation
         │  (B, 46, 16, 16)
         ↓
    Stage 1 CMT Blocks  x2
    ...  (Stages 2–4)
         ↓
    GlobalAvgPool -> Linear Head -> (B, num_classes)

Design rationale
-----------------
Unlike ViT (which uses a single large-stride patch projection) or CvT (which uses an overlapping strided conv with
a large kernel), CMT uses three consecutive 3x3 convolutions:

1.  The first conv (stride=2) halves both spatial dimensions - this is the only spatial downsampling in the stem.
2.  The second and third convs (stride=1) refine the feature representation and extract richer local
    information at that reduced resolution before the first transformer stage.

All convolutions use `GELU -> BatchNorm2d` rather than `LayerNorm` because at this early stage the representation
is still a dense 2-D feature map (not a sequence of tokens).


Spatial output
--------------
For an input of size (H, W):
    - After conv1 (stride=2):  (H/2, W/2)
    - After conv2 (stride=1):  (H/2, W/2)
    - After conv3 (stride=1):  (H/2, W/2)

For Tiny ImageNet (64x64) and CMT-Ti (`stem_channels=32`):
    - input: (B, 3, 64, 64)
    - after conv1: (B, 32, 32, 32)
    - after conv2: (B, 32, 32, 32)
    - Output (after conv3): (B, 32, 32, 32)

"""
import torch
import torch.nn as nn
from torch import Tensor


class CMTStem(nn.Module):
    """
    Three-layer 3x3 convolutional stem for CMT.

    Applies three back-to-back `Conv2d -> GELU -> BatchNorm2d` blocks. The first conv performs 2x spatial downsampling
    (stride=2); the subsequent two layers refine local feature maps at the same resolution and channel size.

    Args:
        in_channels: Number of channels in the input image (3 for RGB).
        stem_channels: Number of output channels for the stem layers.
                       For CMT-Ti and CMT-XS this is 16. For CMT-S it is 32 and for CMT-B it is 38.

    Shape:
        - Input: (B, in_channels, H, W)
        - Output: (B, stem_channels, H/2, W/2)

    Example:
        >>> stem = CMTStem(in_channels=3, stem_channels=32)
        >>> x = torch.randn(2, 3, 64, 64)
        >>> out = stem(x)
        >>> out.shape
        torch.Size([2, 32, 32, 32])
    """

    def __init__(self, in_channels: int, stem_channels: int) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.stem_channels = stem_channels

        # ---- Layer 1: stride-2 conv - spatial downsampling ----
        # (B, in_channels, H, W) -> (B, stem_channels, H/2, W/2)
        self.stem_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=stem_channels, kernel_size=3, stride=2,
                                    padding=1, bias=True)

        # ----- Layer 2: stride-1 conv - local feature extraction ------
        # (B, stem_channels, H/2, W/2) -> (B, stem_channels, H/2, W/2)
        self.stem_conv2 = nn.Conv2d(stem_channels, stem_channels, kernel_size=3, stride=1, padding=1, bias=True)

        # ----- Layer 3: stride-1 conv - further refinement ------
        # (B, stem_channels, H/2, W/2) -> (B, stem_channels, H/2, W/2)
        self.stem_conv3 = nn.Conv2d(stem_channels, stem_channels, kernel_size=3, stride=1, padding=1, bias=True)

        # BatchNorm layers for each conv. Note that the number of channels is `stem_channels` for all three layers.
        self.bn = nn.BatchNorm2d(stem_channels, eps=1e-5)

        # Shared activation - GELU is used throughout CMT (vs. ReLU in plain CNNs).
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Run the three-layer stem on a batch of images.

        Args:
            x: Input image batch of shape (B, in_channels, H, W).

        Returns:
            Feature map of shape (B, stem_channels, H/2, W/2).
        """
        # Conv1: (B, C_in, H, W) -> (B, stem_channels, H/2, W/2)
        x = self.bn(self.activation(self.stem_conv1(x)))

        # Conv2: (B, stem_channels, H/2, W/2) -> (B, stem_channels, H/2, W/2)
        x = self.bn(self.activation(self.stem_conv2(x)))

        # Conv3: (B, stem_channels, H/2, W/2) -> (B, stem_channels, H/2, W/2)
        x = self.bn(self.activation(self.stem_conv3(x)))

        return x

    def output_spatial_size(self, h_in: int, w_in: int) -> tuple[int, int]:
        """
        Compute the output spatial size without running a forward pass.

        Only the first conv applies a stride of 2; the other two preserve spatial dims.

        Args:
            h_in: Input height in pixels.
            w_in: Input width in pixels.

        Returns:
            (H_out, W_out) after the stem.

        Example:
            >>> stem = CMTStem(3, 32)
            >>> stem.output_spatial_size(64, 64)
            (32, 32)
        """
        # Conv1: stride=2, kernel=3, padding=1 -> floor((H + 2 - 3) / 2) + 1 = H // 2
        h_out = (h_in + 2 * 1 - 3) // 2 + 1
        w_out = (w_in + 2 * 1 - 3) // 2 + 1
        # Conv2 & Conv3: stride=1, same padding=1, kernel=3 -> spatial unchanged
        return h_out, w_out

    def extra_repr(self) -> str:
        """Compact summary shown in `print(model)`."""
        return f"in_channels={self.in_channels}, stem_channels (out_channels)={self.stem_channels}"
    