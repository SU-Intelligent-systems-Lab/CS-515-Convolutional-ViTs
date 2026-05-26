"""
Test / evaluation script for ConvViTs.

Follows the structure:
    `get_test_loader`: build the test `DataLoader`.
    `run_eval` : core evaluation loop (no grad, accumulates metrics, logits).
    `run_test` : loads best weights, calls `run_eval`, logs results, and generates all visualizations.

Extensibility
-------------
The file is model-agnostic: `run_test` accepts any `nn.Module` and a `Config`. Adding any architecture in the future
requires no changes here — only `main.py`'s `build_model` needs a new branch.

CvT-specific additions over the base pattern
---------------------------------------------
-   Prediction gallery: image grid with confidence bars.
-   Attention maps: requires `ConvAttention._last_attn` to be set during the forward pass (see `visualization.py`).
"""
from typing import Optional
import os
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from data import build_dataloaders, get_class_name_index_map
from parameters import Config
from utils import (ClassificationMetrics, measure_time, get_logger, extract_attention_weights, plot_attention_maps,
                   plot_prediction_gallery)


logger = get_logger()


def get_test_loader(cfg: Config) -> DataLoader:
    """
    Return the test `DataLoader` for the configured dataset.
    Uses `build_tinyimagenet_dataloaders` and returns only the "test" split so the rest of the test script is
    dataset-agnostic.

    Args:
        cfg: Full `Config` instance.

    Returns:
        `DataLoader` for the held-out test set.
    """
    loaders = build_dataloaders(cfg)
    return loaders["test"]


def run_eval(model: nn.Module, data_loader: DataLoader, cfg: Config, device: torch.device,
             capture_gallery: bool = True) -> dict:
    """
    Core evaluation loop over the full test set.

    Accumulates all metrics, logits, and a sample batch for the prediction gallery. No augmentation is applied.

    Args:
        model: Model in eval mode.
        data_loader: Test `DataLoader`.
        cfg: Full `Config`.
        device: Compute device.
        capture_gallery: If `True`, save the first batch for the prediction gallery visualization.

    Returns:
        Dictionary with keys:

        `metrics`: `dict` from `ClassificationMetrics.compute()`.
        `correct`: int, total correct predictions.
        `n`: int, total samples.
        `all_logits`: (N, C) numpy array.
        `all_labels`: (N, ) numpy array.
        `gallery`: dict with `images`, `logits`, `targets` for the first batch (or `None`).
    """
    metrics = ClassificationMetrics(cfg.data.num_classes, device)

    correct = 0
    n = 0
    all_logits: list[Tensor] = []
    all_labels: list[Tensor] = []
    gallery: Optional[dict] = None

    with torch.no_grad():
        for batch_idx, (imgs, labels) in enumerate(data_loader):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(imgs)
            preds = logits.argmax(dim=1)

            correct += preds.eq(labels).sum().item()
            n += imgs.size(0)

            metrics.update(logits, labels)

            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

            # Capture first batch for prediction gallery
            if capture_gallery and batch_idx == 0:
                gallery = {
                    "images": imgs.cpu(),
                    "logits": logits.cpu(),
                    "targets": labels.cpu(),
                }

    results = metrics.compute()
    logits_np = torch.cat(all_logits).numpy()
    labels_np = torch.cat(all_labels).numpy()

    return {
        "metrics": results,
        "correct": correct,
        "n": n,
        "all_logits": logits_np,
        "all_labels": labels_np,
        "gallery": gallery,
    }


@measure_time
def run_test(model: nn.Module, cfg: Config, device: torch.device, checkpoint_path: Optional[str] = None,
             plot_attention_flag: bool = False) -> dict:
    """
    Load best weights, evaluate on the test set, log and visualize results.

    Mirrors the following structure:
    1. Load the best checkpoint weights into the model.
    2. Build the test `DataLoader`.
    3. Call `run_eval` to collect all metrics and auxiliary outputs.
    4. Log the full result table.
    5. Generate visualizations (prediction gallery, attention maps).

    Args:
        model: Model instance (weights will be overwritten by checkpoint).
        cfg: Full `Config` instance.
        device: Compute device.
        checkpoint_path: Path to a `.pt` checkpoint. Defaults to `<save_dir>/<run_name>/best.pt`.
        plot_attention_flag: Generate per-stage attention heatmaps for the first test image. Requires
                             `ConvAttention._last_attn` to be set during forward (see conv_attention.py).

    Returns:
        `dict` from `run_eval` containing metrics and auxiliary data.
    """
    # Load weights
    if checkpoint_path is None:
        checkpoint_path = os.path.join(cfg.log.save_dir, cfg.log.run_name, "best.pt")

    logger.info(f"Loading weights from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Test set loader
    test_loader = get_test_loader(cfg)

    # Evaluate
    output = run_eval(model, test_loader, cfg, device, capture_gallery=True)

    results = output["metrics"]
    correct = output["correct"]
    n = output["n"]

    logger.info("============ Test Metrics ============")
    logger.info(f"=> Accuracy: {((correct/n)*100):.2f}% ({correct}/{n})")
    logger.info(f"Top-1: {results['top1']:.2f}%, Top-5: {results['top5']:.2f}%, Precision: {results['precision']:.4f}, "
                f"Recall: {results['recall']:.4f}, F1: {results["f1"]:.4f}")
    logger.info("======================================")

    # Per-class accuracy table
    per_cls = results["per_class_acc"] if "per_class_acc" in results else None
    if per_cls is not None:
        logger.info("Per-class accuracy:")
        for i, acc in enumerate(per_cls.tolist()):
            logger.info(f"\tClass {i}: {acc:.2f}")

    model_name = f"{cfg.model.model_name}-{cfg.model.cmt_variant}" \
                 if cfg.model.model_name == "cmt" else cfg.model.model_name

    # Visualizations
    if output["gallery"] is not None:
        logger.info("Generating prediction gallery...")
        g = output["gallery"]
        plot_prediction_gallery(images=g["images"], logits=g["logits"], targets=g["targets"],
                                mean=cfg.data.mean, std=cfg.data.std, model_name=model_name,
                                class_names=get_class_name_index_map(cfg))

    if plot_attention_flag:
        NUM_IMAGES_TO_VISUALIZE = 4
        logger.info(f"Generating attention maps for the first {NUM_IMAGES_TO_VISUALIZE} images...")

        # 1. Fetch the very first batch of data from the loader where images shape: (B, C, H, W), labels shape: (B,)
        images, labels = next(iter(test_loader))

        # Determine how many images to show (cap it at the batch size just in case)
        n_images_to_plot = min(NUM_IMAGES_TO_VISUALIZE, images.size(0))

        # 2. Register the forward hooks onto the attention layers
        attn_store, hooks = extract_attention_weights(model)

        # Expected K/V spatial shapes per stage for CvT (accounting for stride_kv=2)
        kv_spatial_shapes = [(8, 8), (4, 4), (2, 2)]

        # 3. Loop through the images one by one
        for idx in range(n_images_to_plot):
            single_img = images[idx]  # Shape: (C, H, W)

            # Clear the old attention matrix values stored in the dictionary from the previous image
            attn_store.clear()

            # Run a forward pass on just this single image to trigger the hooks
            # We add a batch dimension using unsqueeze(0) -> (1, C, H, W)
            with torch.no_grad():
                model(single_img.unsqueeze(0).to(device))

            # Create a unique filename for each image so they don't overwrite each other
            unique_save_path = f"attention_maps_img{idx}"

            # Plot and save
            plot_attention_maps(
                image=single_img,
                attn_store=attn_store,
                spatial_shapes=kv_spatial_shapes,
                mean=cfg.data.mean,
                std=cfg.data.std,
                save_path=unique_save_path,
            )

        # 4. Clean up the hooks after the loop finishes completely
        for h in hooks:
            h.remove()

    return output
