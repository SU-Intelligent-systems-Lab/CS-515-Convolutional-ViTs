"""Command-line argument parsing and typed configuration dataclasses.

All hyperparameters, paths, and flags live here.  `parse_args()` is the
single entry-point called from `main.py`; it returns a `Config` object
whose fields are typed dataclasses, one per logical concern:

- `ModelConfig`: CvT architecture hyperparameters
- `TrainConfig`: optimizer, scheduler, regularization
- `DataConfig`: dataset paths, augmentation, loaders
- `LogConfig`: logging, checkpointing, W&B / TensorBoard flags

Design notes
------------
- All list-valued model args (`embed_dims`, `depths`, ...) are passed as
  space-separated integers on the CLI and stored as `tuple[int, ...]`.
- Boolean flags use `argparse`'s `BooleanOptionalAction` so both
  `--amp` and `--no-amp` work without a helper function.
- `Config` is itself a frozen dataclass so it can be hashed / used as a
  dict key if needed.
- The `from_namespace` class-method on each sub-config converts the flat
  `argparse.Namespace` into the structured hierarchy.
"""

import argparse
from dataclasses import dataclass, field
from typing import Optional
from utils.logger import _level_from_str


# ---------------------------------------------------------------------------
# Sub-config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DataConfig:
    """
    Dataset, augmentation, and data-loader settings.

    This dataclass is the single source of truth for dataset-specific constants (number of classes, input channels,
    normalization statistics). Extending the project to a new dataset only requires adding entries to the four
    class-level lookup dicts below - no other file needs changing.

    Class-level lookup dicts
    ------------------------
    `MEANS`: per-channel (R, G, B) normalization means.
    `STDS`: per-channel (R, G, B) normalization stds.
    `NUM_CLASSES`: output head size for each dataset identifier.
    `NUM_IN_CHANNELS`: input channel count (3 for all RGB datasets).

    Attributes:
        dataset: Dataset identifier string. Must be a key in the lookup dicts. Defaults to "tiny-imagenet-200".
        data_dir: Root directory of the Tiny ImageNet dataset.
        num_classes: Number of output classes (auto-resolved from `dataset` when not provided explicitly on the CLI).
        in_channels: Number of input image channels (auto-resolved from `dataset`).
        image_size: Spatial size images are resized/cropped to.
        num_workers: Number of `DataLoader` worker processes.
        pin_memory: Pin CPU tensors to GPU memory for faster transfer.
        prefetch_factor: How many batches each worker pre-loads.
        train_split: Fraction of training data used for training (remainder goes to validation when no val split exists)
        use_augmentation: Toggle RandAugment / MixUp augmentation.
        randaugment_n: Number of RandAugment operations per sample.
        randaugment_m: Magnitude of RandAugment operations.
        mixup_alpha: Alpha for Beta distribution in MixUp (0 disables MixUp).
        cutmix_alpha: Alpha for CutMix (0 disables CutMix).
        mean: Per-channel normalization mean (R, G, B).
        std: Per-channel normalization std (R, G, B).
    """
    MEANS: dict[str, tuple[float, ...]] = field(
        default_factory=lambda: {
            "tiny-imagenet-200": (0.4802, 0.4481, 0.3975),
        },
        init=False, repr=False, compare=False,
    )
    STDS: dict[str, tuple[float, ...]] = field(
        default_factory=lambda: {
            "tiny-imagenet-200": (0.2770, 0.2691, 0.2821),
        },
        init=False, repr=False, compare=False,
    )
    NUM_CLASSES: dict[str, int] = field(
        default_factory=lambda: {
            "tiny-imagenet-200": 200,
        },
        init=False, repr=False, compare=False,
    )
    NUM_IN_CHANNELS: dict[str, int] = field(
        default_factory=lambda: {
            "tiny-imagenet-200": 3,
        },
        init=False, repr=False, compare=False,
    )

    dataset: str = "tiny-imagenet-200"
    data_dir: str = f"data_sources/{dataset}"
    num_classes: int = NUM_CLASSES[dataset]
    in_channels: int = NUM_IN_CHANNELS[dataset]
    image_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    train_split: float = 0.9
    use_augmentation: bool = True
    randaugment_n: int = 2
    randaugment_m: int = 9
    mixup_alpha: float = 0.8
    cutmix_alpha: float = 1.0
    mean: tuple[float, ...] = MEANS[dataset]
    std: tuple[float, ...] = STDS[dataset]

    @classmethod
    def from_namespace(cls, ns: argparse.Namespace) -> "DataConfig":
        """
        Construct a `DataConfig` from a parsed `argparse.Namespace`.

        Args:
            ns: Namespace returned by `ArgumentParser.parse_args()`.

        Returns:
            Fully populated `DataConfig` instance.
        """
        return cls(
            dataset=ns.dataset,
            data_dir=ns.data_dir,
            num_classes=ns.num_classes,
            in_channels=ns.in_channels,
            image_size=ns.image_size,
            num_workers=ns.num_workers,
            pin_memory=ns.pin_memory,
            prefetch_factor=ns.prefetch_factor,
            train_split=ns.train_split,
            use_augmentation=ns.use_augmentation,
            randaugment_n=ns.randaugment_n,
            randaugment_m=ns.randaugment_m,
            mixup_alpha=ns.mixup_alpha,
            cutmix_alpha=ns.cutmix_alpha,
            mean=tuple(ns.mean),
            std=tuple(ns.std),
        )


@dataclass
class ModelConfig:
    """
    Architecture hyperparameters for ConvViTs.

    `num_classes` and `in_channels` are kept here so that models can be constructed from a single `ModelConfig`
    without any refactoring. Their values are populated by `parse_args()` from the `DataConfig` registry. They are
    not independent CLI arguments.

    Attributes:
        num_classes: Number of output classes (sourced from `DataConfig`).
        in_channels: Input image channels (sourced from `DataConfig`).
        embed_dims: Token embedding dimension for each of the 3 stages.
        depths: Number of CvT blocks per stage.
        num_heads: Number of attention heads per stage.
        mlp_ratio: Hidden-dim multiplier inside each FFN block.
        qkv_bias: Whether to add a learnable bias to Q/K/V projections.
        drop_rate: Dropout probability on FFN outputs.
        attn_drop_rate: Dropout probability on attention weights.
        drop_path_rate: Maximum stochastic-depth drop probability (linearly increased across all blocks).
        kernel_size_embed: Kernel size for the convolutional token embedding.
        stride_embed: Stride for the convolutional token embedding.
        padding_embed: Padding for the convolutional token embedding.
        kernel_size_proj: Kernel size for the conv Q/K/V projections.
        stride_kv: Stride applied to K and V projections (reduces seq len).
        init_weights: Weight initialization scheme: "trunc_normal" or "kaiming".
    """

    # Sourced from DataConfig - not independent CLI args.
    num_classes: int = 200
    in_channels: int = 3

    # Per-stage settings (3 stages in the original CvT paper)
    embed_dims: tuple[int, ...] = (64, 192, 384)
    depths: tuple[int, ...] = (1, 2, 10)
    num_heads: tuple[int, ...] = (1, 3, 6)
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1

    # Convolutional embedding kernel params
    kernel_size_embed: int = 7
    stride_embed: int = 4
    padding_embed: int = 2

    # Convolutional attention projection params
    kernel_size_proj: int = 3
    stride_kv: int = 2
    init_weights: str = "trunc_normal"

    @classmethod
    def from_namespace(cls, ns: argparse.Namespace) -> "ModelConfig":
        """
        Construct a `ModelConfig` from a parsed `argparse.Namespace`.
        `num_classes` and `in_channels` are read from `ns` where they were already synced from the `DataConfig`
        registry by `parse_args()`.

        Args:
            ns: Namespace returned by `ArgumentParser.parse_args()`.

        Returns:
            Fully populated `ModelConfig` instance.
        """
        return cls(
            num_classes=ns.num_classes,
            in_channels=ns.in_channels,
            embed_dims=tuple(ns.embed_dims),
            depths=tuple(ns.depths),
            num_heads=tuple(ns.num_heads),
            mlp_ratio=ns.mlp_ratio,
            qkv_bias=ns.qkv_bias,
            drop_rate=ns.drop_rate,
            attn_drop_rate=ns.attn_drop_rate,
            drop_path_rate=ns.drop_path_rate,
            kernel_size_embed=ns.kernel_size_embed,
            stride_embed=ns.stride_embed,
            padding_embed=ns.padding_embed,
            kernel_size_proj=ns.kernel_size_proj,
            stride_kv=ns.stride_kv,
            init_weights=ns.init_weights,
        )


@dataclass
class TrainConfig:
    """
    Training loop, optimizer, and scheduler settings.

    Attributes:
        epochs: Total number of training epochs (upper bound when early stopping fires).
        batch_size: Per-GPU training batch size.
        learning_rate: Peak learning rate (after warm-up).
        min_lr: Minimum LR at the end of cosine decay.
        weight_decay: AdamW weight decay.
        beta1: AdamW beta1.
        beta2: AdamW beta2.
        grad_clip: Maximum gradient norm (0 disables clipping).
        warmup_epochs: Number of linear LR warm-up epochs.
        scheduler: LR scheduler type: "cosine" or "step".
        step_size: Epoch step size for `StepLR` (ignored for cosine).
        gamma: LR decay factor for `StepLR` (ignored for cosine).
        amp: Use Automatic Mixed Precision (`torch.cuda.amp`).
        label_smoothing: Label-smoothing epsilon for `CrossEntropyLoss`.
        seed: Global random seed for reproducibility.
        resume: Path to a checkpoint to resume training from.
        profile: Run `ptflops` and exit without training.
        early_stopping_patience: Stop if val loss does not improve for this many consecutive epochs. (0 disables it)
        early_stopping_min_delta: Minimum absolute improvement in val loss that resets the patience counter.
    """
    epochs: int = 300
    batch_size: int = 128
    learning_rate: float = 5e-4
    min_lr: float = 1e-6
    weight_decay: float = 0.05
    beta1: float = 0.9
    beta2: float = 0.999
    grad_clip: float = 1.0
    warmup_epochs: int = 20
    scheduler: str = "cosine"
    step_size: int = 30
    gamma: float = 0.1
    amp: bool = True
    label_smoothing: float = 0.1
    seed: int = 42
    resume: Optional[str] = None
    profile: bool = False
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-4

    @classmethod
    def from_namespace(cls, ns: argparse.Namespace) -> "TrainConfig":
        """
        Construct a `TrainConfig` from a parsed `argparse.Namespace`.

        Args:
            ns: Namespace returned by `ArgumentParser.parse_args()`.

        Returns:
            Fully populated `TrainConfig` instance.
        """
        return cls(
            epochs=ns.epochs,
            batch_size=ns.batch_size,
            learning_rate=ns.learning_rate,
            min_lr=ns.min_lr,
            weight_decay=ns.weight_decay,
            beta1=ns.beta1,
            beta2=ns.beta2,
            grad_clip=ns.grad_clip,
            warmup_epochs=ns.warmup_epochs,
            scheduler=ns.scheduler,
            step_size=ns.step_size,
            gamma=ns.gamma,
            amp=ns.amp,
            label_smoothing=ns.label_smoothing,
            seed=ns.seed,
            resume=ns.resume,
            profile=ns.profile,
            early_stopping_patience=ns.early_stopping_patience,
            early_stopping_min_delta=ns.early_stopping_min_delta,
        )


@dataclass
class LogConfig:
    """
    Logging, checkpointing, and experiment-tracking settings.

    Attributes:
        log_level: Logging verbosity string ("debug", "info", ...).
        log_dir: Directory for log files and TensorBoard event files.
        run_name: Human-readable experiment name (used in file paths).
        save_dir: Directory where model checkpoints are saved.
        save_every: Save a checkpoint every N epochs (0 = only best).
        keep_last: Number of most-recent checkpoints to retain.
        tensorboard: Enable TensorBoard summary writing.
        log_interval: Log training metrics every N batches.
    """
    log_level: str = "info"
    log_dir: str = "logs"
    run_name: str = "exp"
    save_dir: str = "checkpoints"
    save_every: int = 10
    keep_last: int = 3
    tensorboard: bool = False
    log_interval: int = 50

    @classmethod
    def from_namespace(cls, ns: argparse.Namespace) -> "LogConfig":
        """
        Construct a `LogConfig` from a parsed `argparse.Namespace`.

        Args:
            ns: Namespace returned by `ArgumentParser.parse_args()`.

        Returns:
            Fully populated `LogConfig` instance.
        """
        return cls(
            log_level=ns.log_level,
            log_dir=ns.log_dir,
            run_name=ns.run_name,
            save_dir=ns.save_dir,
            save_every=ns.save_every,
            keep_last=ns.keep_last,
            tensorboard=ns.tensorboard,
            log_interval=ns.log_interval,
        )


@dataclass
class Config:
    """
    Top-level container bundling all sub-configs.

    Attributes:
        model: ConvViTs architecture settings.
        train: Optimiser and training-loop settings.
        data: Dataset and augmentation settings.
        log: Logging and checkpointing settings.
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    log: LogConfig = field(default_factory=LogConfig)

    @classmethod
    def from_namespace(cls, ns: argparse.Namespace) -> "Config":
        """
        Build the full `Config` hierarchy from a flat `Namespace`.

        Args:
            ns: Namespace returned by `ArgumentParser.parse_args()`.

        Returns:
            Fully populated `Config` instance.
        """
        return cls(
            model=ModelConfig.from_namespace(ns),
            train=TrainConfig.from_namespace(ns),
            data=DataConfig.from_namespace(ns),
            log=LogConfig.from_namespace(ns),
        )


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """
    Construct the full `ArgumentParser` for the ConvViTs project.

    Groups arguments into four `add_argument_group` sections that mirror the four dataclasses so the `--help`
    output is easy to navigate.

    Returns:
        Configured `ArgumentParser` instance.
    """
    parser = argparse.ArgumentParser(
        description="Train / evaluate ConvViTs on Tiny ImageNet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --------------------------------- Model ---------------------------------
    m = parser.add_argument_group("Model")
    m.add_argument("--embed-dims", type=int, nargs=3, default=[64, 192, 384],
                   metavar=("S1", "S2", "S3"),
                   help="Token embedding dim for each of the 3 stages.")
    m.add_argument("--depths", type=int, nargs=3, default=[1, 2, 10],
                   metavar=("S1", "S2", "S3"),
                   help="Number of CvT blocks per stage.")
    m.add_argument("--num-heads", type=int, nargs=3, default=[1, 3, 6],
                   metavar=("S1", "S2", "S3"),
                   help="Number of attention heads per stage.")
    m.add_argument("--mlp-ratio", type=float, default=4.0,
                   help="FFN hidden-dim multiplier.")
    m.add_argument("--qkv-bias", action=argparse.BooleanOptionalAction,
                   default=True, help="Learnable bias in Q/K/V projections.")
    m.add_argument("--drop-rate", type=float, default=0.0,
                   help="FFN output dropout probability.")
    m.add_argument("--attn-drop-rate", type=float, default=0.0,
                   help="Attention weight dropout probability.")
    m.add_argument("--drop-path-rate", type=float, default=0.1,
                   help="Max stochastic-depth drop probability.")
    m.add_argument("--kernel-size-embed", type=int, default=7,
                   help="Kernel size for convolutional token embedding.")
    m.add_argument("--stride-embed", type=int, default=4,
                   help="Stride for convolutional token embedding.")
    m.add_argument("--padding-embed", type=int, default=2,
                   help="Padding for convolutional token embedding.")
    m.add_argument("--kernel-size-proj", type=int, default=3,
                   help="Kernel size for conv Q/K/V projections.")
    m.add_argument("--stride-kv", type=int, default=2,
                   help="Stride for K and V conv projections.")
    m.add_argument("--init-weights", type=str, default="trunc_normal",
                   choices=["trunc_normal", "kaiming"],
                   help="Weight initialisation scheme.")

    # ----------------------------------------------------------------- Train
    t = parser.add_argument_group("Training")
    t.add_argument("--epochs", type=int, default=300,
                   help="Total training epochs.")
    t.add_argument("--batch-size", type=int, default=128,
                   help="Per-GPU batch size.")
    t.add_argument("--learning-rate", type=float, default=5e-4,
                   help="Peak learning rate.")
    t.add_argument("--min-lr", type=float, default=1e-6,
                   help="Minimum LR at end of cosine decay.")
    t.add_argument("--weight-decay", type=float, default=0.05,
                   help="AdamW weight decay.")
    t.add_argument("--beta1", type=float, default=0.9,
                   help="AdamW beta1.")
    t.add_argument("--beta2", type=float, default=0.999,
                   help="AdamW beta2.")
    t.add_argument("--grad-clip", type=float, default=1.0,
                   help="Max gradient norm (0 disables clipping).")
    t.add_argument("--warmup-epochs", type=int, default=20,
                   help="Linear LR warm-up epochs.")
    t.add_argument("--scheduler", type=str, default="cosine",
                   choices=["cosine", "step"],
                   help="LR scheduler type.")
    t.add_argument("--step-size", type=int, default=30,
                   help="Epoch step size for StepLR.")
    t.add_argument("--gamma", type=float, default=0.1,
                   help="LR decay factor for StepLR.")
    t.add_argument("--amp", action=argparse.BooleanOptionalAction,
                   default=True, help="Use Automatic Mixed Precision.")
    t.add_argument("--label-smoothing", type=float, default=0.1,
                   help="Label-smoothing epsilon.")
    t.add_argument("--seed", type=int, default=42,
                   help="Global random seed.")
    t.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume from.")
    t.add_argument("--profile", action="store_true",
                   help="Run ptflops profiling and exit.")
    t.add_argument("--early-stopping-patience", type=int, default=20,
                   help="Stop if val loss does not improve for N epochs (0 disables early stopping).")
    t.add_argument("--early-stopping-min-delta", type=float, default=1e-4,
                   help="Minimum val-loss improvement to count as progress.")

    # ------------------------------------------------------------------ Data
    d = parser.add_argument_group("Data")
    d.add_argument("--dataset", type=str, default="tiny-imagenet-200",
                   choices=["tiny-imagenet-200"], help="Dataset to use. Controls num_classes, "
                                                       "in_channels, and normalisation defaults")
    d.add_argument("--data-dir", type=str, default="data_sources/tiny-imagenet-200",
                   help="Root directory of the dataset. Defaults to data_sources/<dataset> when omitted.")
    m.add_argument("--num-classes", type=int, default=200,
                   help="Override number of output classes. Defaults to dataset registry value.")
    m.add_argument("--in-channels", type=int, default=3,
                   help="Override input channels. Defaults to dataset registry value.")
    d.add_argument("--image-size", type=int, default=64,
                   help="Spatial size images are resized/cropped to.")
    d.add_argument("--num-workers", type=int, default=4,
                   help="DataLoader worker processes.")
    d.add_argument("--pin-memory", action=argparse.BooleanOptionalAction,
                   default=True, help="Pin tensors to GPU memory.")
    d.add_argument("--prefetch-factor", type=int, default=2,
                   help="Batches each worker pre-loads.")
    d.add_argument("--train-split", type=float, default=0.9,
                   help="Fraction of training data used for training.")
    d.add_argument("--use-augmentation",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Enable RandAugment / MixUp augmentation.")
    d.add_argument("--randaugment-n", type=int, default=2,
                   help="RandAugment number of operations.")
    d.add_argument("--randaugment-m", type=int, default=9,
                   help="RandAugment magnitude.")
    d.add_argument("--mixup-alpha", type=float, default=0.8,
                   help="MixUp alpha (0 disables).")
    d.add_argument("--cutmix-alpha", type=float, default=1.0,
                   help="CutMix alpha (0 disables).")
    d.add_argument("--mean", type=float, nargs=3,
                   default=None,
                   metavar=("R", "G", "B"),
                   help="Override normalization mean. Defaults to dataset registry value.")
    d.add_argument("--std", type=float, nargs=3,
                   default=None,
                   metavar=("R", "G", "B"),
                   help="Override normalization std. Defaults to dataset registry value.")

    # ------------------------------------------------------------------- Log
    lg = parser.add_argument_group("Logging")
    lg.add_argument("--log-level", type=str, default="info",
                    choices=list(("debug", "info", "warning", "error")),
                    help="Logging verbosity.")
    lg.add_argument("--log-dir", type=str, default="logs",
                    help="Directory for log files.")
    lg.add_argument("--run-name", type=str, default="exp",
                    help="Experiment name used in file/dir names.")
    lg.add_argument("--save-dir", type=str, default="checkpoints",
                    help="Directory for model checkpoints.")
    lg.add_argument("--save-every", type=int, default=10,
                    help="Save a checkpoint every N epochs (0 = best only).")
    lg.add_argument("--keep-last", type=int, default=3,
                    help="Number of recent checkpoints to retain.")
    lg.add_argument("--tensorboard", action=argparse.BooleanOptionalAction,
                    default=False, help="Write TensorBoard summaries.")
    lg.add_argument("--log-interval", type=int, default=50,
                    help="Log training metrics every N batches.")

    return parser


def parse_args(argv: Optional[list[str]] = None) -> Config:
    """
    Parse command-line arguments and return a structured `Config`.

    Steps
    -----
    1.  Parse `argv` with the full `ArgumentParser`.
    2.  Sync `num_classes`, `in_channels`, `mean``, `std``, and `data_dir` onto `ns` from the `DataConfig` registry.
        This is the single entry-point where dataset identity propagates to model config without any refactoring
        of model classes.
    3.  Run cross-argument validation.
    4.  Normalize string values and resolve log level.
    5.  Assemble and return a `Config` instance.

    Args:
        argv: Optional list of argument strings. Defaults to `sys.argv[1:]` when `None`.

    Returns:
        Fully validated `Config` instance.

    Raises:
        SystemExit: On `--help` or argument errors (standard argparse behavior).
        ValueError: If cross-argument constraints are violated (e.g. `warmup_epochs >= epochs`).

    Example:
        >>> cfg = parse_args(["--epochs", "100", "--batch-size", "64"])
        >>> cfg.train.epochs
        100
    """
    parser = _build_parser()
    ns = parser.parse_args(argv)

    DEFAULT_DATASET = "tiny-imagenet-200"

    # -------------- resolve dataset-derived defaults --------------
    # Use a temporary DataConfig instance to access the registry dicts.
    _reg = DataConfig()

    if ns.num_classes is None:
        ns.num_classes = _reg.NUM_CLASSES.get(ns.dataset, _reg.NUM_CLASSES[DEFAULT_DATASET])
    if ns.in_channels is None:
        ns.in_channels = _reg.NUM_IN_CHANNELS.get(ns.dataset, _reg.NUM_IN_CHANNELS[DEFAULT_DATASET])
    if ns.mean is None:
        ns.mean = list(_reg.MEANS.get(ns.dataset, _reg.MEANS[DEFAULT_DATASET]))
    if ns.std is None:
        ns.std = list(_reg.STDS.get(ns.dataset, _reg.STDS[DEFAULT_DATASET]))
    if ns.data_dir is None:
        ns.data_dir = f"data_sources/{ns.dataset}"

    # ---------------- Cross-argument validation -----------------
    if ns.warmup_epochs >= ns.epochs:
        parser.error(f"--warmup-epochs ({ns.warmup_epochs}) must be less than --epochs ({ns.epochs}).")
    if not (0.0 < ns.train_split < 1.0):
        parser.error(f"--train-split must be in (0, 1), got {ns.train_split}.")
    if ns.grad_clip < 0.0:
        parser.error(f"--grad-clip must be >= 0, got {ns.grad_clip}.")
    if ns.early_stopping_patience < 0:
        parser.error(f"--early-stopping-patience must be >= 0, got {ns.early_stopping_patience}.")

    # Normalize raw strings so the rest of the code never sees mixed case.
    ns.log_level = ns.log_level.lower()
    ns.scheduler = ns.scheduler.lower()
    ns.init_weights = ns.init_weights.lower()

    # Resolve log level to int early so logger.py doesn't need to import us.
    ns.log_level_int = _level_from_str(ns.log_level)

    return Config.from_namespace(ns)
