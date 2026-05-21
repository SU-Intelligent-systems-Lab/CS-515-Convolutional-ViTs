"""
Visualization utilities for ConvViTs.

Provides plotting functions for training monitoring and model analysis. All functions follow the same convention:
`save_fig` writes to `./assets/` by default, and every function closes its figure after saving to avoid memory leaks.

Functions
---------
`save_fig`: save current figure to disk.
`plot_learning_curves`: loss curves.
`plot_training_dashboard`: multi-panel: loss + accuracy + LR schedule.
`plot_prediction_gallery`: image grid with prediction + confidence bar.
`plot_attention_maps`: CvT attention heatmaps overlaid on image.
"""
import datetime
import os
from typing import Optional
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from utils import get_logger


logger = get_logger()

os.makedirs("./assets/", exist_ok=True)


# ---------------------------- Helpers ----------------------------

def _ts() -> str:
    """Return a filesystem-safe timestamp string."""
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def save_fig(fig_id: str, tight_layout: bool = True, fig_extension: str = "png", resolution: int = 300, 
             assets_path: str = "./assets/") -> None:
    """
    Save the current figure to `<assets_path>/<fig_id>.<ext>`.

    Args:
        fig_id: Filename stem (no extension).
        tight_layout: Call `plt.tight_layout()` before saving.
        fig_extension: File format (default "png").
        resolution: DPI for raster formats (default 300).
        assets_path: Output directory (created if absent).
    """
    os.makedirs(assets_path, exist_ok=True)
    path = os.path.join(assets_path, f"{fig_id}.{fig_extension}")
    logger.info("Saving figure -> %s", path)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution, bbox_inches="tight")


# ---------------------- Plotting Functions -----------------------

def plot_learning_curves(history: dict, save_path: str = f"loss_curves_{_ts()}") -> None:
    """
    Plot training vs. validation loss curves.

    Args:
        history: Dict with keys "train_loss" and "val_loss" (lists of floats, one per epoch).
        save_path: Filename stem passed to `save_fig`.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, history["train_loss"], label="Training Loss", linewidth=2)
    plt.plot(epochs, history["val_loss"], label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    # plt.grid(True, alpha=0.3)

    save_fig(save_path)
    plt.close()


def plot_training_dashboard(history: dict, save_path: str = f"training_dashboard_{_ts()}") -> None:
    """
    Multi-panel training overview: loss, top-1, top-5, and LR schedule.

    Args:
        history: Dict with any subset of keys: "train_loss", "val_loss", "train_top1", "val_top1", "val_top5", "lr".
        save_path: Filename stem.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("CvT Training Dashboard", fontsize=16, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32)

    # Loss
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(epochs, history["train_loss"], label="Train", linewidth=2)
    ax.plot(epochs, history["val_loss"], label="Val", linewidth=2)
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Top-1 accuracy
    ax = fig.add_subplot(gs[0, 1])
    if "train_top1" in history:
        ax.plot(epochs, history["train_top1"], label="Train Top-1", linewidth=2)
    if "val_top1" in history:
        ax.plot(epochs, history["val_top1"], label="Val Top-1", linewidth=2)
    ax.set_title("Top-1 Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Top-5 accuracy
    ax = fig.add_subplot(gs[1, 0])
    if "train_top5" in history:
        ax.plot(epochs, history["train_top5"], label="Train Top-1", linewidth=2)
    if "val_top5" in history:
        ax.plot(epochs, history["val_top5"], label="Val Top-5", linewidth=2)
    ax.set_title("Val Top-5 Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # LR schedule
    ax = fig.add_subplot(gs[1, 1])
    if "lr" in history:
        ax.semilogy(epochs, history["lr"], color="darkorange", linewidth=2)
        ax.set_title("Learning Rate (log scale)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("LR")
        ax.grid(True, alpha=0.3, which="both")
    else:
        ax.set_visible(False)

    save_fig(save_path, tight_layout=False)
    plt.close(fig)


def plot_prediction_gallery(images: Tensor, logits: Tensor, targets: Tensor, mean: tuple[float, ...],
                            std: tuple[float, ...], class_names: Optional[list[str]] = None, n_cols: int = 6,
                            n_rows: int = 4, save_path: str = f"prediction_gallery_{_ts()}") -> None:
    """
    Image grid showing predictions with colored borders and confidence bars.

    Green border = correct prediction.
    Red border = wrong prediction.
    Each cell title shows true label (True:) and predicted label (P:). A small bar below shows the max softmax
    confidence as a percentage.

    Args:
        images: (N, C, H, W) normalized float tensor.
        logits: (N, num_classes) float tensor.
        targets: (N, ) long tensor of ground-truth labels.
        class_names: Optional class name list.
        mean: Channel-wise mean vector used to de-normalize the input image.
        std: Channel-wise standard deviation vector used to de-normalize the image.
        n_cols: Grid columns.
        n_rows: Grid rows.
        save_path: Destination filename string.
    """
    n_show = min(n_cols * n_rows, images.size(0))
    images = images[:n_show].cpu()
    logits = logits[:n_show].cpu()
    targets = targets[:n_show].cpu()

    # Calculate probabilities and predictions
    probs = torch.softmax(logits, dim=1)
    confs, preds = probs.max(dim=1)

    # De-normalize images safely
    _mean = torch.tensor(mean).view(3, 1, 1)
    _std = torch.tensor(std).view(3, 1, 1)
    imgs_disp = (images * _std + _mean).clamp(0, 1)

    def lbl(idx: int) -> str:
        return class_names[idx][:16] if class_names else str(idx)

    # Expanded vertical spacing slightly (from 3.2 to 3.6) to make room for confidence bars
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.4, n_rows * 3.6))
    fig.suptitle("Prediction Gallery (✓ correct  /  ✗ wrong)", fontsize=14, fontweight="bold")
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i >= n_show:
            ax.axis("off")
            continue

        img = imgs_disp[i].permute(1, 2, 0).numpy()
        pred = preds[i].item()
        tgt = targets[i].item()
        conf = confs[i].item()
        correct = (pred == tgt)
        col = "#2ecc71" if correct else "#e74c3c"
        sym = "✓" if correct else "✗"

        # Display image
        ax.imshow(img)

        # Clear ticks/labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        # Configure custom colorful prediction bounding box spines
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(col)
            spine.set_linewidth(3.5)

        ax.set_title(f"{sym} True: {lbl(tgt)}\n  Predicted: {lbl(pred)}", fontsize=9, color=col, pad=4, fontweight="semibold")

        # Anchor the confidence bar context safely inside subplot footprint
        # Shifted location indices safely inside the subplot boundaries
        ax_ins = ax.inset_axes([0.0, -0.16, 1.0, 0.10])
        ax_ins.barh(0, conf * 100, color=col, height=1.0)
        ax_ins.set_xlim(0, 100)
        ax_ins.axis("off")

        # Dynamically place text: center of bar if high confidence, right of bar if low confidence
        text_x = conf * 50 if conf > 0.3 else 50
        text_col = "white" if conf > 0.3 else col

        ax_ins.text(text_x, 0, f"{conf * 100:.0f}%", ha="center", va="center", fontsize=7.5, color=text_col,
                    fontweight="bold")

    plt.subplots_adjust(hspace=0.45, wspace=0.3)
    save_fig(save_path, tight_layout=False)
    plt.close(fig)
    logger.info(f"Prediction gallery saved -> {save_path}.png")


def plot_attention_maps(image: Tensor, attn_store: dict[str, Tensor], spatial_shapes: list[tuple[int, int]],
                        mean: tuple[float, ...], std:  tuple[float, ...],
                        save_path: str = f"attention_maps", add_ts: bool = True) -> None:
    """
    Overlay CvT attention heatmaps on the input image for each stage.
    For each architectural stage, averages attention weights across all heads and query positions, maps the 1-D
    sequence back into its 2-D spatial (H_kv, W_kv) feature layout, upscales it to match the input resolution,
    and overlays it semi-transparently.
    This explicitly visualizes "What spatial regions does CvT attend to at each resolution scale?"

    Layout:
        One row per stage. Columns: [Original Image | Raw Heatmap | Translucent Overlay]

    Args:
        image: Single input image tensor of shape (C, H, W), normalized.
        attn_store: Dictionary mapping module names to attention tensors of shape (1, heads, N_q, N_kv) captured via
                    forward hooks.
        spatial_shapes: List of expected (H, W) spatial configurations per stage, e.g., [(16, 16), (8, 8), (4, 4)].
        mean: Channel-wise mean vector used to de-normalize the input image.
        std: Channel-wise standard deviation vector used to de-normalize the image.
        save_path: Destination filename string.
        add_ts: Flag to add timestamp to the destination filename
    """
    if not attn_store:
        logger.warning("attn_store is empty — skipping attention map plot.")
        return

    # 1. De-normalize and prepare image for visualization
    _m = torch.tensor(mean).view(3, 1, 1)
    _s = torch.tensor(std).view(3, 1, 1)
    img_disp = (image.cpu() * _s + _m).clamp(0, 1).permute(1, 2, 0).numpy()
    H, W = img_disp.shape[:2]

    # 2. Structural Stage Grouping via Module Key Names
    n_stages = len(spatial_shapes)
    stage_groups: list[list[Tensor]] = [[] for _ in range(n_stages)]

    # Dynamic string dispatch avoids collapsing asymmetrical block distributions
    # (e.g., Stage 1 having 1 block while Stage 3 has 10 blocks).
    for name, attn in sorted(attn_store.items()):
        if "stage1" in name:
            si = 0
        elif "stage2" in name:
            si = 1
        elif "stage3" in name:
            si = 2
        else:
            logger.info(f"Ignoring non-stage block target: {name}")
            continue

        # Reduce attention: (1, heads, N_q, N_kv) -> mean over heads and query positions
        # This yields a pure 1-D array of length (N_kv,) reflecting local spatial contexts
        block_avg = attn[0].mean(dim=0).mean(dim=0)
        stage_groups[si].append(block_avg)

    # 3. Consolidate and Reshape Tensors Into Spatial Map Matrices
    stage_hmaps: list[np.ndarray] = []
    for si, group in enumerate(stage_groups):
        if not group:
            logger.warning(f"No attention weights found matching Stage {si + 1}")
            continue

        # Safely stack blocks since they are structurally guaranteed to be uniform length now
        avg = torch.stack(group).mean(dim=0)  # (N_kv,)
        h, w = spatial_shapes[si]
        n_kv = avg.numel()

        # Handle projection padding discrepancies safely if spatial targets diverge slightly
        if n_kv == h * w:
            hmap = avg.reshape(h, w).numpy()
        else:
            side = int(np.sqrt(n_kv))
            hmap = avg[:side * side].reshape(side, side).numpy()
            logger.info(f"Stage {si + 1} size mismatch layout: fell back to square {side}x{side}.")

        stage_hmaps.append(hmap)

    if not stage_hmaps:
        logger.warning("No valid attention map matrices generated. Aborting plot.")
        return

    # 4. Render and Export Multi-Stage Heatmaps
    n_rows = len(stage_hmaps)
    fig, axes = plt.subplots(n_rows, 3, figsize=(12, 4 * n_rows))

    # Force 2D index continuity if evaluating a single-stage model slice
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle("CvT Attention Heatmaps per Stage", fontsize=14, fontweight="bold")

    for si, hmap in enumerate(stage_hmaps):
        h, w = hmap.shape

        # Min-max scaling for vibrant visibility contrast
        hmap_n = (hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-8)

        # Upscale the miniature latent maps to input canvas size via pixel-exact nearest-neighbor Kronecker product
        scale_h = max(1, H // h)
        scale_w = max(1, W // w)
        hmap_up = np.kron(hmap_n, np.ones((scale_h, scale_w)))[:H, :W]

        # Column 0: Clean reference source image
        axes[si, 0].imshow(img_disp)
        axes[si, 0].set_title(f"Stage {si + 1} — Input ({h}x{w} tokens)", fontsize=9)
        axes[si, 0].axis("off")

        # Column 1: Isolated low-res raw attention array matrix (Pure Heatmap)
        axes[si, 1].imshow(hmap, cmap="hot", interpolation="bicubic")
        axes[si, 1].set_title("Attn Weights\n(head-avg, query-avg)", fontsize=9)
        axes[si, 1].axis("off")

        # Column 2: Blended image canvas showing alpha heatmap weights (With Overlay)
        axes[si, 2].imshow(img_disp)
        axes[si, 2].imshow(hmap_up, cmap="jet", alpha=0.45, interpolation="bicubic", vmin=0, vmax=1)
        axes[si, 2].set_title("Overlay (α=0.45)", fontsize=9)
        axes[si, 2].axis("off")

    if add_ts:
        save_path = f"{save_path}_{_ts()}"

    save_fig(save_path, tight_layout=False)
    plt.close(fig)
    logger.info(f"Attention maps saved -> {save_path}.png")


def extract_attention_weights(model: torch.nn.Module) -> tuple[dict[str, Tensor], list]:
    """
    Register forward hooks on all `ConvAttention` modules.

    Args:
        model: CvT model instance.

    Returns:
        (attn_store, hooks): `attn_store` is filled after a forward pass; remove hooks with [h.remove() for h in hooks].
    """
    attn_store: dict[str, Tensor] = {}
    hooks: list = []

    def _make_hook(name: str):
        def hook(module, inp, out):
            if hasattr(module, "_last_attn"):
                attn_store[name] = module._last_attn.detach().cpu()
        return hook

    for name, module in model.named_modules():
        if type(module).__name__ == "ConvAttention":
            hooks.append(module.register_forward_hook(_make_hook(name)))
    return attn_store, hooks
