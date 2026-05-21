"""
Batch-level MixUp and CutMix augmentation.

Applied after the `DataLoader` collates a batch: not inside `Dataset.__getitem__`: because both methods require
operating on pairs of samples simultaneously.

Usage in the training loop:
    aug = MixUpCutMix(
        num_classes=cfg.data.num_classes,
        mixup_alpha=cfg.data.mixup_alpha,
        cutmix_alpha=cfg.data.cutmix_alpha,
    )

    for images, labels in train_loader:
        images, soft_labels = aug(images.to(device), labels.to(device))
        logits = model(images)
        loss   = criterion(logits, soft_labels)

Both methods convert integer hard labels to soft (one-hot mixed) targets so the same `CrossEntropyLoss` call works
for all three cases: no-aug (standard one-hot), MixUp, and CutMix.

References:
    * MixUp: Zhang et al. (2018) https://arxiv.org/abs/1710.09412
    * CutMix: Yun et al. (2019) https://arxiv.org/abs/1905.04899
"""
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


class MixUpCutMix:
    """
    Batch-level MixUp and CutMix augmentation.

    For each batch, randomly selects between MixUp and CutMix (or neither) based on which methods are enabled.
    When both are enabled, one is chosen uniformly at random.

    Setting either alpha to 0 disables the corresponding method. When both are 0 the module is effectively a
    no-op (returns one-hot labels for API consistency so the training loop need not branch).

    Args:
        num_classes: Number of output classes (needed to build one-hot encodings).
        mixup_alpha: Beta-distribution concentration for MixUp. 0 disables MixUp.
        cutmix_alpha: Beta-distribution concentration for CutMix. 0 disables CutMix.

    Example:
        >>> aug = MixUpCutMix(num_classes=200, mixup_alpha=0.8, cutmix_alpha=1.0)
        >>> images = torch.randn(32, 3, 64, 64)
        >>> labels = torch.randint(0, 200, (32,))
        >>> mixed_images, soft_labels = aug(images, labels)
        >>> mixed_images.shape
        torch.Size([32, 3, 64, 64])
        >>> soft_labels.shape
        torch.Size([32, 200])
    """

    def __init__(self, num_classes: int, mixup_alpha: float = 0.8, cutmix_alpha: float = 1.0) -> None:
        self.num_classes = num_classes
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self._use_mixup = mixup_alpha > 0.0
        self._use_cutmix= cutmix_alpha > 0.0

    def __call__(self, images: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
        """
        Apply MixUp or CutMix (or neither) to a training batch.

        Args:
            images: Float tensor (B, C, H, W) on any device.
            labels: Long tensor (B, ) with integer class indices, on the same device as `images`.

        Returns:
            A tuple (mixed_images, soft_labels) where:
                - `mixed_images`: (B, C, H, W) float tensor.
                - `soft_labels`: (B, num_classes) float tensor; ows are convex combinations of two one-hot vectors.
        """
        if not (self._use_mixup or self._use_cutmix):
            return images, self._one_hot(labels)

        use_cutmix = self._use_cutmix and (not self._use_mixup or random.random() > 0.5)
        if use_cutmix:
            return self._cutmix(images, labels)
        return self._mixup(images, labels)

    # ---------------------- MixUp ----------------------

    def _mixup(self, images: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
        """
        Linearly blend pairs of images and their labels.

        Samples λ ~ Beta(α, α) and computes:
            mixed_image = λ · img_i + (1-λ) · img_j
            soft_label  = λ · onehot_i + (1-λ) · onehot_j
        where j is a uniformly random permutation of batch indices.

        Args:
            images: (B, C, H, W) float tensor.
            labels: (B, ) long tensor.

        Returns:
            (mixed_images, soft_labels): both batch-size B.
        """
        lam: float = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
        B = images.size(0)
        idx = torch.randperm(B, device=images.device)

        mixed = lam * images + (1.0 - lam) * images[idx]
        soft = lam * self._one_hot(labels) + (1.0 - lam) * self._one_hot(labels[idx])

        return mixed, soft

    # --------------------- CutMix ----------------------

    def _cutmix(self, images: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
        """
        Paste a random rectangular patch from one image into another.

        Samples λ ~ Beta(α, α), derives a bounding box whose area is approximately (1-λ) · H · W, pastes the patch
        from a random partner image, then recomputes λ from the actual box area:
            actual_λ = 1  -  (box_area / (H · W))
            soft_label = actual_λ · onehot_i  +  (1-actual_λ) · onehot_j

        Args:
            images: (B, C, H, W) float tensor.
            labels: (B, ) long tensor.

        Returns:
            (mixed_images, soft_labels): both batch-size B.
        """
        lam = float(np.random.beta(self.cutmix_alpha, self.cutmix_alpha))
        B, _, H, W = images.shape
        idx = torch.randperm(B, device=images.device)

        x1, y1, x2, y2 = self._rand_bbox(H, W, lam)

        # Recompute lambda from the actual clipped box area.
        lam = 1.0 - (x2 - x1) * (y2 - y1) / (H * W)

        mixed = images.clone()
        mixed[:, :, y1:y2, x1:x2] = images[idx, :, y1:y2, x1:x2]

        soft = lam * self._one_hot(labels) + (1.0 - lam) * self._one_hot(labels[idx])

        return mixed, soft

    @staticmethod
    def _rand_bbox(H: int, W: int, lam: float) -> tuple[int, int, int, int]:
        """
        Sample a random bounding box for CutMix.

        The box dimensions are chosen so that its area fraction is approximately (1 - lam).

        Args:
            H: Image height in pixels.
            W: Image width in pixels.
            lam: Lambda sampled from the Beta distribution.

        Returns:
            (x1, y1, x2, y2) corner coordinates clipped to image bounds.
        """
        cut_ratio = np.sqrt(1.0 - lam)
        cut_h = int(H * cut_ratio)
        cut_w = int(W * cut_ratio)

        cx = random.randint(0, W)
        cy = random.randint(0, H)

        x1 = max(cx - cut_w // 2, 0)
        y1 = max(cy - cut_h // 2, 0)
        x2 = min(cx + cut_w // 2, W)
        y2 = min(cy + cut_h // 2, H)

        return x1, y1, x2, y2

    def _one_hot(self, labels: Tensor) -> Tensor:
        """
        Convert integer class labels to one-hot float vectors.

        Args:
            labels: (B, ) long tensor with class indices in [0, num_classes).

        Returns:
            (B, num_classes) float tensor.
        """
        return F.one_hot(labels, self.num_classes).float()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"num_classes={self.num_classes}, "
            f"mixup_alpha={self.mixup_alpha}, "
            f"cutmix_alpha={self.cutmix_alpha})"
        )
