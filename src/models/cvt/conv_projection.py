"""
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

"""
import torch.nn as nn
from torch import Tensor


class ConvProjection(nn.Module):
    """
    Depthwise separable convolution that projects tokens into Q, K, or V.

    The module reshapes the flat token sequence back into a 2-D feature map. Then it applies a depthwise conv
    (one filter per channel) followed by BatchNorm and GELU, then a pointwise 1x1 conv to mix channels, and
    finally flattens back to a sequence.

    The spatial shape after projection is:  `H_out = floor((H + 2 x pad - kernel_size) / stride) + 1`

    For Q  : stride = 1,         so  H_out = H   (sequence length unchanged)
    For K,V: stride = stride_kv, so  H_out < H   (sequence length reduced)

    Args:
        in_channels: Number of input embedding channels C.
        out_channels: Number of output embedding channels C'. Usually equal to `in_channels`.
        kernel_size: Spatial kernel size for the depthwise conv.
        stride: Spatial stride. Use 1 for Q, `stride_kv` for K and V.
        padding: Zero-padding for the depthwise conv. Typically, `kernel_size // 2` to preserve spatial size
                 when stride = 1.
        bias: Whether to add bias to the pointwise conv.

    Shape:
        - Input:  `(B, N, C)` + spatial hint `(H, W)`
        - Output: `(B, H'*W', C')` + updated `(H', W')`
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int,
                 bias: bool = False) -> None:
        super().__init__()
        self.stride = stride
        self.dw_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,  # This makes the conv depthwise. `groups` will change kernel shape into (in_channels, 1, k, k).
            bias=False,
        )
        self.bn = nn.BatchNorm2d(in_channels)
        self.activation = nn.GELU()
        self.pw_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=bias,
        )

    def forward(self, x: Tensor, h: int, w: int) -> tuple[Tensor, int, int]:
        """
        Project a token sequence through the depthwise separable conv.

        Args:
            x: Token sequence of shape `(B, N, C)` where `N = H * W`.
            h: Current spatial height `H`.
            w: Current spatial width `W`.

        Returns:
            A tuple `(out, h_out, w_out)` where
            - `out`: projected tokens of shape `(B, N', C')`
            - `h_out`: output spatial height
            - `w_out`: output spatial width
        """
        B, N, C = x.shape

        # 1. Token Sequence -> 2-D token map: (B, N, C) -> (B, C, H, W)
        x = x.transpose(1, 2).reshape(B, C, h, w)

        # 2. Depthwise conv -> BN -> GELU -> Pointwise 1x1 conv
        x = self.dw_conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.pw_conv(x)

        # 3. Record new spatial dims before flattening.
        h_out, w_out = x.shape[2], x.shape[3]

        # 4. Reshape 2-D feature map into sequence again: (B, C', H', W') -> (B, H'*W', C')
        x = x.flatten(2).transpose(1, 2)

        return x, h_out, w_out

    def extra_repr(self) -> str:
        """Compact parameter summary for `print(model)`."""
        return (
            f"dw_kernel={self.dw_conv.kernel_size[0]}, "
            f"stride={self.stride}, "
            f"in={self.dw_conv.in_channels}, "
            f"out={self.pw_conv.out_channels}"
        )
