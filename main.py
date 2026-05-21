"""
Entry point for training and evaluation.

Follows the structure:
    `set_seed` : fix all random sources for reproducibility.
    `main` : device detection, seed, model build, FLOPs, run.

Extensibility
-------------
Adding a new architecture requires only:

1. Implementing the model in `models/new_architecture/`.
2. Adding one `elif` branch in `build_model` in models/__init__.py.
3. Adding any new CLI flags to `parameters.py`.

Nothing in `train.py`, `test.py`, or `evaluation.py` changes.

Modes
-----
- `--mode train`: run training only.
- `--mode test`: load best checkpoint and run test evaluation.
- `--mode both`: train then immediately evaluate on test set.
- `--profile`: print FLOPs + param count and exit.
"""
import logging
import random
import datetime
import os
import numpy as np
import torch
from utils import compute_flops, setup_logger
from models import build_model
from parameters import parse_args
from test import run_test
from train import run_training


# ------------------- Reproducibility -------------------

def set_seed(seed: int) -> None:
    """
    Fix all random seeds for reproducibility.

    Applies the same seed to Python's `random`, NumPy, PyTorch CPU, and all CUDA devices. Also sets cuDNN to
    deterministic mode. Note that deterministic mode can slow down training slightly. Disable with
    `torch.backends.cudnn.deterministic = False` if speed is critical and exact reproducibility is not required.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------ Device Selection -------------------
def _select_device() -> torch.device:
    """
    Select the best available compute device. Priority: CUDA -> MPS (Apple Silicon) -> CPU.

    Returns:
        `torch.device` for the selected backend.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def main() -> None:
    """Parse arguments, set up the run, and dispatch to train / test."""
    # Parse args -> Config
    cfg = parse_args()

    # setting up logger
    log_file = os.path.join(cfg.log.log_dir, cfg.log.run_name,
                            f"run_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log")
    setup_logger(level=cfg.log.log_level_int, log_file=log_file)

    # Re-get logger after setup so this module's logger inherits the config
    logger = logging.getLogger("convvits")

    logger.info("=" * 40)
    logger.info(f"ConvViTs - run: {cfg.log.run_name}")
    logger.info(f"Dataset: {cfg.data.dataset}  ({cfg.data.num_classes} classes)")
    logger.info(f"Epochs: {cfg.train.epochs}, batch={cfg.train.batch_size},  lr={cfg.train.learning_rate:.2e}")
    logger.info(f"AMP: {cfg.train.amp}, grad_clip={cfg.train.grad_clip:.1f}")
    logger.info(f"Scheduler: {cfg.train.scheduler}, warmup={cfg.train.warmup_epochs}")
    logger.info(f"Early stop: patience={cfg.train.early_stopping_patience}, "
                f"delta={cfg.train.early_stopping_min_delta:.1e}")
    logger.info("=" * 40)
    logger.info(f"Full Arguments: {cfg}")
    logger.info("=" * 40)

    # Seed
    set_seed(cfg.train.seed)
    logger.info(f"Seed set to {cfg.train.seed}")

    # Select Device
    device = _select_device()
    logger.info(f"Using device: {device}")

    # Build model
    model = build_model(cfg)
    logger.info(f"Model: {cfg.model.model_name} - "
                f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} - "
                f"Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    logger.info(model)

    # Analyze FLOPs + params
    flops, params_count = compute_flops(model,
                                        input_size=(cfg.data.in_channels, cfg.data.image_size, cfg.data.image_size))
    logger.info(f"Model complexity: FLOPs: {flops} | Params: {params_count}")

    # Mode dispatch
    mode = getattr(cfg.train, "mode", "both")   # default: Both = train + test
    if mode in ("train", "both"):
        logger.info("Starting training...")
        model = run_training(model, cfg, device)

    if mode in ("test", "both"):
        logger.info("Starting test evaluation...")
        run_test(
            model=model,
            cfg=cfg,
            device=device,
            checkpoint_path=cfg.train.resume,   # None -> auto best.pt
            plot_attention_flag=True
        )

    # Exit after profiling if --profile flag is set
    if mode == "profile":
        logger.info("--profile flag set. Exiting after complexity analysis.")
        return


if __name__ == "__main__":
    main()
