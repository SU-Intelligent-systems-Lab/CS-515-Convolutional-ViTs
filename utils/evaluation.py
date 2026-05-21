"""
Classification metrics and FLOPs profiling for ConvViTs.

This module only knows about measuring model quality and complexity.

Implements all classification metrics using `torchmetrics`: the community standard for correct, numerically stable
metric computation in PyTorch. It handles edge cases (zero-division per class, distributed accumulation) that
hand-rolled implementations often miss.
"""
import torch
import torch.nn as nn
from torch import Tensor
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)
from ptflops import get_model_complexity_info


# ---------------- Computing Classification Metrics -----------------

class ClassificationMetrics:
    """
    Stateful accumulator for multi-class classification metrics.
    Wraps `torchmetrics` metrics with the same `update / compute / reset` interface used in the training loop.
    All metrics are accumulated over an entire epoch or test set and computed together at the end.

    Metrics computed
    ----------------
    - Top-1 accuracy: percentage of samples where argmax prediction matches the true label.
    - Top-5 accuracy: percentage of samples where true label appears in the top-5 predicted classes.
    - Macro precision: mean of per-class TP / (TP + FP), unweighted by class frequency.
    - Macro recall: mean of per-class TP / (TP + FN).
    - Macro F1: harmonic mean of macro precision and recall.
    - Confusion matrix: (C, C) integer tensor where rows = true/actual, cols = predicted.

    Args:
        num_classes: Number of output classes.
        device: Device on which metric tensors are accumulated. Should match the device used in the forward pass.
    """
    def __init__(self, num_classes: int, device: torch.device | str) -> None:
        self.num_classes = num_classes
        self.device = torch.device(device)

        top_k = min(5, num_classes)

        self._top1 = MulticlassAccuracy(num_classes=num_classes, top_k=1, average="micro").to(self.device)

        self._top5 = MulticlassAccuracy(num_classes=num_classes, top_k=top_k, average="micro").to(self.device)

        self._precision = MulticlassPrecision(num_classes=num_classes, average="macro").to(self.device)

        self._recall = MulticlassRecall(num_classes=num_classes, average="macro").to(self.device)

        self._f1 = MulticlassF1Score(num_classes=num_classes, average="macro").to(self.device)

        self._cm = MulticlassConfusionMatrix(num_classes=num_classes).to(self.device)

    def _all_metrics(self):
        return [self._top1, self._top5, self._precision, self._recall, self._f1, self._cm]

    def reset(self) -> None:
        """Reset all metric accumulators. Call at the start of each epoch."""
        for metric in self._all_metrics():
            metric.reset()

    def update(self, logits: Tensor, targets: Tensor) -> None:
        """
        Accumulate predictions for one batch.

        Args:
            logits: Raw model output (B, num_classes): softmax is applied internally by torchmetrics.
            targets: Ground-truth integer labels (B,).
        """
        logits = logits.detach().to(self.device)
        targets = targets.detach().to(self.device)

        for metric in self._all_metrics():
            metric.update(logits, targets)

    def compute(self) -> dict:
        """
        Compute all metrics from accumulated state.

        Returns:
            Dictionary with keys:
            - `top1` : float, top-1 accuracy (0–100).
            - `top5` : float, top-5 accuracy (0–100).
            - `precision` : float, macro precision (0–1).
            - `recall` : float, macro recall (0–1).
            - `f1` : float, macro F1 (0–1).
            - `confusion_matrix`: (C, C) long tensor on CPU.
        """
        # torchmetrics returns values in [0, 1]. We should multiply by 100 for getting percentages.
        return {
            "top1": self._top1.compute().item() * 100.0,
            "top5": self._top5.compute().item() * 100.0,
            "precision": self._precision.compute().item(),
            "recall": self._recall.compute().item(),
            "f1": self._f1.compute().item(),
            "confusion_matrix": self._cm.compute().cpu(),
        }

    def __repr__(self) -> str:
        return (
            f"ClassificationMetrics("
            f"num_classes={self.num_classes}, "
            f"device={self.device})"
        )


# -------------------- FLOPs/parameter profiling --------------------

def compute_flops(model: nn.Module, input_size: tuple[int, int, int] = (3, 64, 64),
                  print_per_layer: bool = False) -> tuple[str, str]:
    """
    Compute FLOPs and parameter count using `ptflops`.

    Wraps `get_model_complexity_info` with sensible defaults for the ConvViTs project. The model is set to eval mode
    before profiling and restored to its original mode after.

    Args:
        model: PyTorch model to profile.
        input_size: (C, H, W) input shape (without batch dimension). Defaults to `(3, 64, 64)` for Tiny ImageNet.
        print_per_layer: If `True`, print per-layer FLOPs breakdown to stdout.

    Returns:
        Tuple (flops_str, params_str): human-readable strings such as "4.23 GMac" and "19.98 M".
    """
    was_training = model.training
    model.eval()

    try:
        flops, params = get_model_complexity_info(
            model,
            input_size,
            as_strings=True,
            print_per_layer_stat=print_per_layer,
            verbose=False,
        )
    finally:
        model.train(was_training)   # always restore original mode
    return flops, params
