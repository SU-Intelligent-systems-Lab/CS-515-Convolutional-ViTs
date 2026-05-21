"""
Training loop for ConvViTs.

Main Structure:
    `train_one_epoch`: one full pass over the training set.
    `validate`: evaluation on the validation set.
    `run_training`: outer loop: calls both, handles scheduling, checkpointing, and early stopping.


Adaptations for CvT beyond the base pattern
--------------------------------------------
-   AdamW + cosine LR with linear warm-up: standard for vision transformers; warm-up prevents early instability from
    large LR on untrained weights.
-   Automatic Mixed Precision (AMP): halves VRAM usage and speeds up training on Ampere+ GPUs with negligible accuracy
    impact.
-   Gradient clipping: clips global L2 norm before each optimizer step, preventing occasional large gradient spikes
    in deep transformers.
-   MixUp / CutMix: applied per batch after moving to device; soft labels require a soft cross-entropy loss function.
-   Checkpoint management: saves the best checkpoint by val loss and optionally periodic snapshots, pruning old ones
    to `keep_last`.

All training utilities (`EarlyStopping`, `AverageMeter`, `get_optimizer`, `build_scheduler`, checkpoint helpers) live
here alongside the training loop so `train.py` is self-contained.
"""
import copy
import dataclasses
import glob
import os
from math import inf
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from data import build_tinyimagenet_dataloaders
from data.mixup_cutmix import MixUpCutMix
from parameters import Config
from utils import measure_time, get_logger, ClassificationMetrics
from utils.visualization import plot_learning_curves, plot_training_dashboard


logger = get_logger()


# ---------------------------- Early Stopping ----------------------------

class EarlyStopping:
    """
    Stop training when validation loss stops improving. Extended with `min_delta` to avoid halting on negligible
    fluctuations and `state_dict` / `load_state_dict` for checkpoint resumption.

    Args:
        patience: Epochs to wait without improvement before setting `self.stop = True` (0 disables early stopping).
        min_delta: Minimum absolute decrease in val loss that counts as an improvement and resets the patience counter.

    Attributes:
        stop: Set to `True` when patience is exhausted.
        best_loss: Best validation loss seen so far.
        counter: Number of consecutive epochs without improvement.
    """
    def __init__(self, patience: int = 15, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss: float = inf
        self.counter: int = 0
        self.stop: bool = False

    def step(self, val_loss: float) -> None:
        """
        Update state with the latest validation loss.

        Args:
            val_loss: Validation loss for the current epoch.
        """
        if self.patience == 0:
            return

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            logger.info(f"EarlyStopping: no improvement {self.counter}/{self.patience} (best={self.best_loss:.6f})")
            if self.counter >= self.patience:
                self.stop = True
                logger.info("EarlyStopping triggered.")

    def state_dict(self) -> dict:
        """Serializable state for checkpoint saving."""
        return {
            "counter": self.counter,
            "best_loss": self.best_loss,
            "stop": self.stop,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore state from a checkpoint dict."""
        self.counter = state["counter"]
        self.best_loss = state["best_loss"]
        self.stop = state["stop"]

    def __repr__(self) -> str:
        return f"EarlyStopping(patience={self.patience}, counter={self.counter}, best={self.best_loss:.6f})"


# ----------------------------- Average Meter ----------------------------

class AverageMeter:
    """
    Running mean tracker: one instance per scalar (loss, acc, time).

    Args:
        name: Human-readable name for `__repr__`.
        fmt: Format string for `val` and `avg` in `__str__`.
    """

    def __init__(self, name: str = "", fmt: str = ":f") -> None:
        self.name = name
        self.fmt = fmt
        self.val: float = 0.0
        self.avg: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """Accumulate `val` over n samples."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        fmtstr = f"{{name}} {{val{self.fmt}}} ({{avg{self.fmt}}})"
        return fmtstr.format(self.__dict__)

    def __repr__(self) -> str:
        return f"AverageMeter(name={self.name!r}, avg={self.avg:.4f}, n={self.count})"


# ------------------------------ Optimizers ------------------------------

def get_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    """
    Build an AdamW optimizer from `TrainConfig`.
    Only parameters with `requires_grad=True` are passed so frozen layers (if any) are excluded automatically.

    Args:
        model: ConvViT model.
        cfg: Full `Config` instance.

    Returns:
        Configured `AdamW` optimizer.
    """
    trainable = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(
        trainable,
        lr=cfg.train.learning_rate,
        betas=(cfg.train.beta1, cfg.train.beta2),
        weight_decay=cfg.train.weight_decay,
    )
    logger.info(f"Optimizer: AdamW(lr={cfg.train.learning_rate:.2e}, wd={cfg.train.weight_decay:.2e}, "
                f"betas=({cfg.train.beta1:.2f}, {cfg.train.beta2:.3f}))")
    return optimizer


# ------------------------------ Schedulers ------------------------------

def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Config) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Build the LR scheduler with optional linear warm-up.

    - Warm-up phase: For the first `warmup_epochs` epochs, LR scales linearly from `min_lr` to `learning_rate`. This
    prevents unstable early iterations when weights are randomly initialized.

    - After warm-up
        - "cosine": `CosineAnnealingLR` decaying to `min_lr`.
        - "step" : `StepLR` with `step_size` and `gamma`.

    Both are combined with the warm-up via `SequentialLR`.

    Args:
        optimizer: The optimizer whose LR will be scheduled.
        cfg: Full `Config` instance.

    Returns:
        A `torch.optim.lr_scheduler.LRScheduler` instance.
    """
    tc = cfg.train

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=tc.min_lr / max(tc.learning_rate, 1e-12),
        end_factor=1.0,
        total_iters=tc.warmup_epochs,
    )

    if tc.scheduler == "cosine":
        main = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=tc.epochs - tc.warmup_epochs,
            eta_min=tc.min_lr,
        )
    elif tc.scheduler == "step":
        main = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=tc.step_size,
            gamma=tc.gamma,
        )
    else:
        raise ValueError(f"Unknown scheduler: {tc.scheduler}")

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, main],
        milestones=[tc.warmup_epochs],
    )
    logger.info(f"Scheduler: {tc.warmup_epochs} epoch warm-up -> {tc.scheduler}")
    return scheduler


# Soft cross-entropy  (needed when MixUp / CutMix returns float targets)

def soft_cross_entropy(logits: Tensor, soft_targets: Tensor) -> Tensor:
    """
    Cross-entropy loss that accepts soft (float) target distributions.

    Used when MixUp / CutMix is active and targets are convex combinations
    of one-hot vectors rather than integer labels.

    Args:
        logits: (B, C) raw model output.
        soft_targets: (B, C) float probability distribution (rows sum to 1.0).

    Returns:
        Scalar mean cross-entropy loss.
    """
    log_probs = F.log_softmax(logits, dim=1)
    return -(soft_targets * log_probs).sum(dim=1).mean()


# ------------------------ Checkpoint Management -------------------------

def save_checkpoint(state: dict, save_dir: str, filename: str, keep_last: int = 3) -> None:
    """
    Save a checkpoint and prune old ones.

    Args:
        state: Dictionary with `model_state_dict`, `optimizer_state_dict`, `epoch`, etc.
        save_dir: Directory for checkpoint files.
        filename: Checkpoint filename (e.g. "epoch_042.pt").
        keep_last: Maximum number of epoch checkpoints to retain. Best checkpoints ("best.pt") are never pruned.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(save_dir, filename)
    torch.save(state, path)
    logger.info(f"Checkpoint saved -> {path}")

    # Prune old epoch checkpoints (keep best.pt untouched)
    if "best" not in filename:
        epoch_ckpts = sorted(glob.glob(os.path.join(save_dir, "epoch_*.pt")))
        while len(epoch_ckpts) > keep_last:
            os.remove(epoch_ckpts.pop(0))


def load_checkpoint(path: str, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, scheduler=None,
                    early_stopping: Optional[EarlyStopping] = None, device: torch.device = torch.device("cpu")) -> int:
    """
    Load a checkpoint and restore all stateful objects.

    Args:
        path: Path to the `.pt` checkpoint file.
        model: Model to restore weights into.
        optimizer: Optional optimizer to restore state into.
        scheduler: Optional LR scheduler to restore.
        early_stopping: Optional `EarlyStopping` to restore.
        device: Device to map checkpoint tensors onto.

    Returns:
        `epoch`: the epoch at which the checkpoint was saved, so training loop can resume from the correct iteration.
    """
    logger.info(f"Loading checkpoint from {path}")
    ckpt = torch.load(path, map_location=device)

    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    if early_stopping is not None and "early_stopping_state" in ckpt:
        early_stopping.load_state_dict(ckpt["early_stopping_state"])

    epoch = ckpt.get("epoch", 0)
    logger.info(f"Resumed from epoch {epoch} (best_val_loss={ckpt.get("best_val_loss", float("inf")):.4f})")
    return epoch


# --------------------------- Model Validation ---------------------------

def validate(model: nn.Module, loader: torch.utils.data.DataLoader, criterion: nn.Module, device: torch.device,
             cfg: Config, epoch: int) -> tuple[float, float, float]:
    """
    Evaluate the model on the validation set.
    No augmentation is applied. Logs macro precision, recall, and F1 via `ClassificationMetrics`.

    Args:
        model: ConvViT model.
        loader: Validation `DataLoader`.
        criterion: `CrossEntropyLoss`.
        device: Compute device.
        cfg: Full `Config`.
        epoch: Current epoch (for logging).

    Returns:
        (mean_val_loss, top1_acc, top5_acc) over the full val set.
    """
    model.eval()
    loss_meter = AverageMeter("val_loss", ":.4f")
    val_metrics = ClassificationMetrics(cfg.data.num_classes, device)

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc=f"Epoch {epoch} [val]", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(imgs)
            loss = criterion(logits, labels)

            loss_meter.update(loss.item(), n=imgs.size(0))
            val_metrics.update(logits, labels)

    results = val_metrics.compute()

    logger.info(f"{'=' * 10} Validation Metrics {'=' * 10}")
    logger.info(f"Epoch {epoch} => loss={loss_meter.avg:.4f}, Top-1={results['top1']:.4f}%, "
                f"Top-5={results['top5']:.4f}%, Precision={results['precision']:.4f}, "
                f"Recall={results['recall']:.4f}, F1={results['f1']:.4f}")
    logger.info("=" * 40)
    return loss_meter.avg, results["top1"], results["top5"]


# ---------------------------- Model Training ----------------------------

def train_one_epoch(model: nn.Module, loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer,
                    augmenter: Optional[MixUpCutMix], device: torch.device, cfg: Config, scaler: GradScaler,
                    epoch: int) -> tuple[float, float, float]:
    """
    Run one full training epoch.

    Steps per batch
    ---------------
    1. Move images and labels to device.
    2. Apply MixUp / CutMix (batch-level, after device transfer) if augmentation is enabled.
    3. Forward pass (inside `autocast` when AMP is enabled).
    4. Compute loss: soft CE when augmenter is active, label-smoothing CE otherwise.
    5. Backward + gradient clip + optimizer step (via `GradScaler`).
    6. Accumulate loss, top-1, and top-5 accuracy meters.

    Args:
        model: ConvViT model in training mode.
        loader: Training `DataLoader`.
        optimizer: Configured optimizer.
        augmenter: `MixUpCutMix` instance or `None`.
        device: Compute device.
        cfg: Full `Config`.
        scaler: AMP `GradScaler` (identity-like when AMP is disabled).
        epoch: Current epoch number (for tqdm description).

    Returns:
        `(mean_loss, top1_accuracy, top5_accuracy)` over the epoch.
    """
    model.train()
    loss_meter = AverageMeter("loss", ":.4f")
    acc1_meter = AverageMeter("top1", ":.4f")
    acc5_meter = AverageMeter("top5", ":.4f")
    use_amp = cfg.train.amp and device.type == "cuda"

    # Standard CE with label smoothing for the no-augmentation path
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing)

    progress_bar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False)

    for imgs, labels in progress_bar:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        B = imgs.size(0)

        #  MixUp / CutMix
        if augmenter is not None and cfg.data.use_augmentation:
            imgs, soft_labels = augmenter(imgs, labels)
            use_soft = True
        else:
            use_soft = False

        optimizer.zero_grad(set_to_none=True)

        # Forward
        with autocast("cuda", enabled=use_amp):
            logits = model(imgs)
            if use_soft:
                loss = soft_cross_entropy(logits, soft_labels)
            else:
                loss = criterion(logits, labels)

        # Backward + clip + step
        scaler.scale(loss).backward()

        if cfg.train.grad_clip > 0.0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        # Metrics Calculation (Optimized for Top-1 & Top-5)
        with torch.no_grad():
            # 1. Fetch top 5 predictions along the class dimension. -> maxk_indices shape: (B, 5)
            _, maxk_indices = logits.topk(5, dim=1, largest=True, sorted=True)

            # 2. Reshape original labels to match: (B, 1) -> expands to (B, 5) during comparison
            # correct shape: (B, 5) with True values where predictions matched targets
            correct = maxk_indices.eq(labels.view(-1, 1))

            # Top-1 accuracy checks just the first column (the absolute argmax)
            top1 = correct[:, 0].float().mean().item() * 100.0

            # Top-5 accuracy checks if the true label appears anywhere across all 5 column slots
            top5 = correct.any(dim=1).float().mean().item() * 100.0

        loss_meter.update(loss.item(), n=B)
        acc1_meter.update(top1, n=B)
        acc5_meter.update(top5, n=B)

        # Update bar overlay display tracking metrics live
        progress_bar.set_postfix(loss=f"{loss_meter.avg:.4f}", top1=f"{acc1_meter.avg:.4f}%",
                                 top5=f"{acc5_meter.avg:.4f}%")

    return loss_meter.avg, acc1_meter.avg, acc5_meter.avg


@measure_time
def run_training(model: nn.Module, cfg: Config, device: torch.device) -> nn.Module:
    """
    Full training loop.

    New features adapted for ConvViTs:
    - AdamW + cosine scheduler with linear warm-up.
    - Automatic Mixed Precision (AMP) `GradScaler` for mixed-precision training.
    - Gradient clipping per batch.
    - MixUp / CutMix applied inside the epoch loop.
    - Best checkpoint saved by val loss; periodic epoch checkpoints pruned to `keep_last`.
    - `EarlyStopping.step()` + `early_stopping.stop` check after each epoch.

    Args:
        model: Initialized ConvViT model (not yet on device).
        cfg: Parsed `Config` instance.
        device: Compute device (`cuda` or `cpu`).

    Returns:
        Model loaded with the best weights found during training.
    """
    model = model.to(device)

    # Getting Data
    loaders = build_tinyimagenet_dataloaders(cfg)
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    # Augmentation
    augmenter = (
        MixUpCutMix(
            num_classes=cfg.data.num_classes,
            mixup_alpha=cfg.data.mixup_alpha,
            cutmix_alpha=cfg.data.cutmix_alpha,
        )
        if cfg.data.use_augmentation else None
    )

    # Optimizer and scheduler
    optimizer = get_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    # Hard-label CE for validation (no Augmentation on val set)
    val_criterion = nn.CrossEntropyLoss()

    # Automatic Mixed Precision scaler
    use_amp = cfg.train.amp and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)

    # Early stopping
    early_stopping = (
        EarlyStopping(
            patience=cfg.train.early_stopping_patience,
            min_delta=cfg.train.early_stopping_min_delta,
        )
        if cfg.train.early_stopping_patience > 0 else None
    )

    # Resume from checkpoint
    start_epoch = 0
    if cfg.train.resume:
        start_epoch = load_checkpoint(cfg.train.resume, model, optimizer, scheduler, early_stopping, device)

    # Training state
    best_val_loss = inf
    best_val_top1 = 0.0
    best_weights = None

    history: dict[str, list] = {
        "train_loss": [], "val_loss": [],
        "train_top1": [], "val_top1": [],
        "train_top5": [], "val_top5": [],
        "lr": [],
    }
    save_dir = os.path.join(cfg.log.save_dir, cfg.log.run_name)

    # Epoch loop
    for epoch in range(start_epoch + 1, cfg.train.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{cfg.train.epochs}")

        # Train
        tr_loss, tr_top1, tr_top5 = train_one_epoch(model, train_loader, optimizer, augmenter, device, cfg, scaler,
                                                    epoch)
        logger.info(f"=> Training loss: {tr_loss:.4f} - Training Top-1 Accuracy: {tr_top1:.4f} - "
                    f"Training Top-5 Accuracy: {tr_top5:.4f}")

        # Validate
        val_loss, val_top1, val_top5 = validate(model, val_loader, val_criterion, device, cfg, epoch)
        logger.info(f"=> Validation loss: {val_loss:.4f} - Validation Top-1 Accuracy: {val_top1:.4f} - "
                    f"Validation Top-5 Accuracy: {val_top5}")

        # LR step
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"=> Learning Rate: {current_lr:.4e}")

        # Log to history
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_top1"].append(tr_top1)
        history["val_top1"].append(val_top1)
        history["train_top5"].append(tr_top5)
        history["val_top5"].append(val_top5)
        history["lr"].append(current_lr)

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_top1 = val_top1
            best_weights = copy.deepcopy(model.state_dict())    # snapshot in memory
            save_checkpoint(
                state={
                    "epoch": epoch,
                    "model_state_dict": best_weights,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "early_stopping_state": (early_stopping.state_dict() if early_stopping else {}),
                    "best_val_loss": best_val_loss,
                    "best_val_top1": best_val_top1,
                    "history": history,
                    "cfg": dataclasses.asdict(cfg),
                },
                save_dir=save_dir,
                filename="best.pt",
                keep_last=cfg.log.keep_last,
            )
            logger.info(f"Best model updated (val_loss={best_val_loss:.4f}, val_top1={best_val_top1:.4f}%)")

        # Periodic epoch checkpoint
        if cfg.log.save_every > 0 and epoch % cfg.log.save_every == 0:
            save_checkpoint(
                state={
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "early_stopping_state": (early_stopping.state_dict() if early_stopping else {}),
                    "best_val_loss": best_val_loss,
                    "best_val_top1": best_val_top1,
                    "history": history,
                    "cfg": cfg,
                },
                save_dir=save_dir,
                filename=f"epoch_{epoch:03d}.pt",
                keep_last=cfg.log.keep_last,
            )

        # Early stopping
        if early_stopping is not None:
            early_stopping.step(val_loss)
            if early_stopping.stop:
                logger.warning(f"Early stopping triggered. Epoch {epoch - cfg.train.early_stopping_patience} had "
                               f"the lowest validation loss ({best_val_loss})")
                break

    # Restore best weights into the model before returning
    if best_weights is not None:
        model.load_state_dict(best_weights)
        logger.info(f"Training complete. Best val_loss={best_val_loss:.4f}, val_top-1={best_val_top1:.4f}%")

    # Save plots
    plot_learning_curves(history)
    plot_training_dashboard(history)

    return model
