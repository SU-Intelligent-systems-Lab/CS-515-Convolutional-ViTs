"""
# FFN — Feed-Forward Network (MLP) related to the Convolutional Transformer Block

"""
import torch
import torch.nn as nn
from torch import Tensor


class CvTFFN(nn.Module):
    """
    Position-wise Feed-Forward Network (MLP) applied to each token.

    A two-layer fully-connected network applied independently to each token in the sequence. The intermediate dimension
    is expanded by `mlp_ratio` relative to the embedding dimension, giving the model capacity to learn complex
    per-token transformations while keeping cross-token interaction exclusively within the attention submodule.

    Architecture:
        1. Linear(C, C x mlp_ratio)
        2. GELU
        3. Dropout(drop)
        4. Linear(C x mlp_ratio, C)
        5. Dropout(drop)

    Args:
        in_features: Input and output embedding dimension C.
        mlp_ratio: Hidden layer expansion factor. Default (4.0) means the hidden layer is 4x wider than the embedding.
        drop: Dropout probability applied after each linear layer.

    Shape:
        - Input: (B, N, C)
        - Output: (B, N, C) - shape is unchanged.

    Example:
        >>> ffn = CvTFFN(in_features=64, mlp_ratio=4.0, drop=0.1)
        >>> x = torch.randn(2, 256, 64)
        >>> ffn(x).shape
        torch.Size([2, 256, 64])
    """
    def __init__(self, in_features: int, mlp_ratio: float = 4.0, drop: float = 0.0) -> None:
        super().__init__()

        hidden = int(in_features * mlp_ratio)

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, in_features),
            nn.Dropout(drop),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply the two-layer MLP to every token independently.

        Args:
            x: Token sequence of shape (B, N, C).

        Returns:
            Transformed token sequence of shape (B, N, C).
        """
        return self.net(x)

    def extra_repr(self) -> str:
        fc1 = self.net[0]  # first Linear
        return (
            f"in={fc1.in_features}, "
            f"hidden={fc1.out_features}, "
            f"out={self.net[3].out_features}"
        )
