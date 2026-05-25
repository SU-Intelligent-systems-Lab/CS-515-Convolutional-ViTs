"""
Command-line argument parsing and typed configuration dataclasses.

All hyperparameters, paths, and flags live here. `parse_args()` is the single entry-point called from `main.py`, it
returns a `Config` object whose fields are typed dataclasses, one per logical concern:

- `ModelConfig`: ConvViT architecture hyperparameters
- `TrainConfig`: optimizer, scheduler, regularization
- `DataConfig`: dataset paths, augmentation, loaders
- `LogConfig`: logging, checkpointing, W&B / TensorBoard flags

Design notes
------------
-   All list-valued model args (`embed_dims`, `depths`, ...) are passed as space-separated integers on the CLI and
    stored as `tuple[int, ...]`.
-   Boolean flags use argparse's `BooleanOptionalAction` so both `--amp` and `--no-amp` work without a helper function.
-   `Config` is itself a frozen dataclass so it can be hashed / used as a dict key if needed.
-   The `from_namespace` class-method on each sub-config converts the flat `argparse.Namespace` into the
    structured hierarchy.
-   CMT variant presets (Ti / XS / S / B) are stored as a class-level registry in `ModelConfig`. Passing
    `--cmt-variant ti` on the CLI auto-fills all hyperparameters.
"""

import argparse
from dataclasses import dataclass, field
from typing import Optional
from utils.logger import _level_from_str


# --------------------------------------------
# Sub-config dataclasses
# --------------------------------------------


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
    data_dir: str = field(init=False)
    num_classes: int = field(init=False)
    in_channels: int = field(init=False)
    mean: tuple[float, ...] = field(init=False)
    std: tuple[float, ...] = field(init=False)

    def __post_init__(self):
        """Calculate dependent fields after the instance is initialized."""
        # Ensure the requested dataset exists in configurations
        if self.dataset not in self.NUM_CLASSES:
            raise ValueError(f"Dataset '{self.dataset}' is not configured in DataConfig lookups.")

        # Safely compute dependent values based on the assigned dataset
        self.data_dir = f"data_sources/{self.dataset}"
        self.num_classes = self.NUM_CLASSES[self.dataset]
        self.in_channels = self.NUM_IN_CHANNELS[self.dataset]
        self.mean = self.MEANS[self.dataset]
        self.std = self.STDS[self.dataset]

    @classmethod
    def from_namespace(cls, ns: argparse.Namespace) -> "DataConfig":
        """
        Construct a `DataConfig` from a parsed `argparse.Namespace`.
        Note: Since data_dir, mean, std, etc., are handled by __post_init__, we only need to pass the base overrides
        from our CLI namespace.

        Args:
            ns: Namespace returned by `ArgumentParser.parse_args()`.

        Returns:
            Fully populated `DataConfig` instance.
        """
        return cls(
            dataset=ns.dataset,
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
        )


@dataclass
class ModelConfig:
    """
    Architecture hyperparameters for ConvViTs.

    Class-level `CMT_VARIANTS` lookup dict: maps shorthand keys ("ti", "xs", "s", "b") to the corresponding paper
                                            hyperparameters.
        `stem_channels`: Number of output channels of the stem.
        `channel_dims`: Per-stage channel dimensions. Each stage's `PatchEmbedding` maps the previous stage's `channel_dims`
                       to the next.
        `depths`: Number of CMT Blocks per stage.
        `mlp_ratios`: Per-stage IRFFN hidden-dim expansion factor.

    Attributes:
        model_name: Model identifier string. Must be a key in the lookup (_MODEL_REGISTRY) dict. Defaults to "cvt".
        cvt_embed_dims: Token embedding dimension for each of the 3 stages of CvT.
        cvt_depths: Number of CvT blocks per stage.
        cvt_num_heads: Number of attention heads per stage.
        cvt_mlp_ratio: Hidden-dim multiplier inside each FFN block.
        cvt_kernel_size_embed: Kernel size for the convolutional token embedding.
        cvt_stride_embed: Stride for the convolutional token embedding.
        cvt_padding_embed: Padding for the convolutional token embedding.
        cvt_kernel_size_proj: Kernel size for the conv Q/K/V projections.
        cvt_stride_kv: Stride applied to K and V projections (reduces seq len).
        qkv_bias: Whether to add a learnable bias to Q/K/V CvT Conv. projections or CMT linear projections
                  inside LMHSA.
        drop_rate: Dropout probability on CvT FFN outputs or CMT IRFFN outputs.
        attn_drop_rate: Dropout probability on attention weights.
        drop_path_rate: Maximum stochastic-depth drop probability (linearly increased across all blocks).
        cmt_variant: Shorthand name of the preset used for CMT ("ti", "xs", "s", "b").
        init_weights: Weight initialization scheme: "trunc_normal" or "kaiming".
    """
    CMT_VARIANTS: dict[str, dict] = field(
        default_factory=lambda: {
            "ti": {
                "stem_channels": 16,
                "channel_dims": (46, 92, 184, 368),
                "depths": (2, 2, 10, 2),
                "mlp_ratios": (3.6, 3.6, 3.6, 3.6),
            },
            "xs": {
                "stem_channels": 16,
                "channel_dims": (52, 104, 208, 416),
                "depths": (3, 3, 12, 3),
                "mlp_ratios": (3.8, 3.8, 3.8, 3.8),
            },
            "s": {
                "stem_channels": 32,
                "channel_dims": (64, 128, 256, 512),
                "depths": (3, 3, 16, 3),
                "mlp_ratios": (4.0, 4.0, 4.0, 4.0),
            },
            "b": {
                "stem_channels": 38,
                "channel_dims": (76, 152, 304, 608),
                "depths": (4, 4, 20, 4),
                "mlp_ratios": (4.0, 4.0, 4.0, 4.0),
            }
        },
        init=False, repr=False, compare=False,
    )

    model_name: str = "cvt"

    # Per-stage settings (3 stages in the original CvT paper)
    cvt_embed_dims: tuple[int, ...] = (64, 192, 384)
    cvt_depths: tuple[int, ...] = (1, 2, 10)
    cvt_num_heads: tuple[int, ...] = (1, 3, 6)
    cvt_mlp_ratio: float = 4.0

    # Convolutional embedding kernel params
    cvt_kernel_size_embed: int = 7
    cvt_stride_embed: int = 4
    cvt_padding_embed: int = 2

    # Convolutional attention projection params
    cvt_kernel_size_proj: int = 3
    cvt_stride_kv: int = 2

    qkv_bias: bool = True
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1

    cmt_variant: str = "ti"

    init_weights: str = "trunc_normal"

    cmt_stem_channels: int = field(init=False)
    cmt_channel_dims: tuple[int, ...] = field(init=False)
    cmt_depths: tuple[int, ...] = field(init=False)
    cmt_mlp_ratios: tuple[float, ...] = field(init=False)

    def __post_init__(self):
        """Calculate dependent fields after the instance is initialized."""
        # Ensure the requested dataset exists in configurations
        if self.cmt_variant.lower() not in self.CMT_VARIANTS.keys():
            raise ValueError(f"Unknown CMT Variant:'{self.cmt_variant.lower()}'. "
                             f"Valid options: {list(self.CMT_VARIANTS.keys())}")

        # Safely compute dependent values based on the assigned dataset
        self.cmt_stem_channels = self.CMT_VARIANTS[self.cmt_variant.lower()]["stem_channels"]
        self.cmt_channel_dims = self.CMT_VARIANTS[self.cmt_variant.lower()]["channel_dims"]
        self.cmt_depths = self.CMT_VARIANTS[self.cmt_variant.lower()]["depths"]
        self.cmt_mlp_ratios = self.CMT_VARIANTS[self.cmt_variant.lower()]["mlp_ratios"]


    @classmethod
    def from_namespace(cls, ns: argparse.Namespace) -> "ModelConfig":
        """
        Construct a `ModelConfig` from a parsed `argparse.Namespace`.

        Args:
            ns: Namespace returned by `ArgumentParser.parse_args()`.

        Returns:
            Fully populated `ModelConfig` instance.
        """
        return cls(
            model_name=ns.model_name,
            cvt_embed_dims=tuple(ns.cvt_embed_dims),
            cvt_depths=tuple(ns.cvt_depths),
            cvt_num_heads=tuple(ns.cvt_num_heads),
            cvt_mlp_ratio=ns.cvt_mlp_ratio,
            qkv_bias=ns.qkv_bias,
            drop_rate=ns.drop_rate,
            attn_drop_rate=ns.attn_drop_rate,
            drop_path_rate=ns.drop_path_rate,
            cvt_kernel_size_embed=ns.cvt_kernel_size_embed,
            cvt_stride_embed=ns.cvt_stride_embed,
            cvt_padding_embed=ns.cvt_padding_embed,
            cvt_kernel_size_proj=ns.cvt_kernel_size_proj,
            cvt_stride_kv=ns.cvt_stride_kv,
            cmt_variant=ns.cmt_variant,
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
        mode: Possible Values ["train", "test", "both", "profile"]. if "profile", run `ptflops` and exit without
              training.
        early_stopping_patience: Stop if val loss does not improve for this many consecutive epochs (0 disables it).
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
    mode: str = None
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
            mode=ns.mode,
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
        log_interval: Log training metrics every N batches.
    """
    log_level: str = "info"
    log_dir: str = "logs"
    run_name: str = "exp"
    save_dir: str = "checkpoints"
    save_every: int = 10
    keep_last: int = 3
    log_interval: int = 50
    log_level_int: int = field(init=False)

    def __post_init__(self):
        self.log_level_int = _level_from_str(self.log_level.lower())

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
            log_interval=ns.log_interval,
        )


@dataclass
class Config:
    """
    Top-level container bundling all sub-configs.
    Both models CvT and CMT are always populated regardless of which model is selected. The active model is determined
    by `model.model_name`.

    Attributes:
        model: ConvViTs architecture settings.
        train: Optimizer and training-loop settings.
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


# --------------------------------------------
# Argument parser
# --------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """
    Construct the full `ArgumentParser` for the ConvViTs project.

    Groups arguments into four `add_argument_group` sections: Model architecture, Training, Data, and Logging.

    Returns:
        Configured `ArgumentParser` instance.
    """
    parser = argparse.ArgumentParser(
        description="Train / evaluate ConvViTs on Tiny ImageNet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --------------------------------- Model ---------------------------------
    m = parser.add_argument_group("Model")
    m.add_argument("--model-name", type=str, default="cvt", choices=["cvt", "cmt"],
                   help="Model to use.")
    m.add_argument("--cvt-embed-dims", type=int, nargs=3, default=[64, 192, 384],
                   metavar=("S1", "S2", "S3"),
                   help="CvT Token embedding dim for each of the 3 stages.")
    m.add_argument("--cvt-depths", type=int, nargs=3, default=[1, 2, 10],
                   metavar=("S1", "S2", "S3"),
                   help="CvT Number of blocks per stage.")
    m.add_argument("--cvt-num-heads", type=int, nargs=3, default=[1, 3, 6],
                   metavar=("S1", "S2", "S3"),
                   help="CvT Number of attention heads per stage.")
    m.add_argument("--cvt-mlp-ratio", type=float, default=4.0,
                   help="CvT FFN hidden-dim multiplier.")
    m.add_argument("--cvt-kernel-size-embed", type=int, default=7,
                   help="CvT Kernel size for convolutional token embedding.")
    m.add_argument("--cvt-stride-embed", type=int, default=4,
                   help="CvT Stride for convolutional token embedding.")
    m.add_argument("--cvt-padding-embed", type=int, default=2,
                   help="CvT Padding for convolutional token embedding.")
    m.add_argument("--cvt-kernel-size-proj", type=int, default=3,
                   help="CvT Kernel size for conv Q/K/V projections.")
    m.add_argument("--cvt-stride-kv", type=int, default=2,
                   help="CvT Stride for K and V conv projections.")
    m.add_argument("--qkv-bias", action=argparse.BooleanOptionalAction,
                   default=True, help="Learnable bias in Q/K/V projections.")
    m.add_argument("--drop-rate", type=float, default=0.0,
                   help="FFN output dropout probability.")
    m.add_argument("--attn-drop-rate", type=float, default=0.0,
                   help="Attention weight dropout probability.")
    m.add_argument("--drop-path-rate", type=float, default=0.05,
                   help="Max stochastic-depth drop probability.")
    m.add_argument("--init-weights", type=str, default="trunc_normal",
                   choices=["trunc_normal", "kaiming"],
                   help="Weight initialisation scheme.")
    m.add_argument("--cmt-variant", type=str, default="ti", choices=["ti", "xs", "s", "b"],
                   help="CMT variants presented in the paper: ti (tiny), xs (extra-small), s (small), b (base).")

    # ------------------------------- Training --------------------------------
    t = parser.add_argument_group("Training")
    t.add_argument("--epochs", type=int, default=300, help="Total training epochs.")
    t.add_argument("--batch-size", type=int, default=128, help="Per-GPU batch size.")
    t.add_argument("--learning-rate", type=float, default=7e-4, help="Peak learning rate.")
    t.add_argument("--min-lr", type=float, default=1e-5, help="Minimum LR at end of cosine decay.")
    t.add_argument("--weight-decay", type=float, default=0.05, help="AdamW weight decay.")
    t.add_argument("--beta1", type=float, default=0.9, help="AdamW beta1.")
    t.add_argument("--beta2", type=float, default=0.999, help="AdamW beta2.")
    t.add_argument("--grad-clip", type=float, default=1.0,
                   help="Max gradient norm (0 disables clipping).")
    t.add_argument("--warmup-epochs", type=int, default=10, help="Linear LR warm-up epochs.")
    t.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "step"],
                   help="LR scheduler type.")
    t.add_argument("--step-size", type=int, default=30, help="Epoch step size for StepLR.")
    t.add_argument("--gamma", type=float, default=0.1, help="LR decay factor for StepLR.")
    t.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                   help="Use Automatic Mixed Precision.")
    t.add_argument("--label-smoothing", type=float, default=0.1, help="Label-smoothing epsilon.")
    t.add_argument("--seed", type=int, default=42, help="Global random seed.")
    t.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume from.")
    t.add_argument("--mode", type=str, default="both", choices=["train", "test", "both", "profile"],
                   help="Execution mode.")
    t.add_argument("--early-stopping-patience", type=int, default=40,
                   help="Stop if val loss does not improve for N epochs (0 disables early stopping).")
    t.add_argument("--early-stopping-min-delta", type=float, default=1e-4,
                   help="Minimum val-loss improvement to count as progress.")

    # --------------------------------- Data ----------------------------------
    d = parser.add_argument_group("Data")
    d.add_argument("--dataset", type=str, default="tiny-imagenet-200",
                   choices=["tiny-imagenet-200"],
                   help="Dataset to use. Controls num_classes, in_channels, and normalisation defaults.")
    d.add_argument("--data-dir", type=str, default=None,
                   help="Root directory of the dataset. Defaults to data_sources/<dataset> when omitted.")
    d.add_argument("--image-size", type=int, default=64,
                   help="Spatial size images are resized/cropped to.")
    d.add_argument("--num-workers", type=int, default=4,
                   help="DataLoader worker processes.")
    d.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True,
                   help="Pin tensors to GPU memory.")
    d.add_argument("--prefetch-factor", type=int, default=2,
                   help="Batches each worker pre-loads.")
    d.add_argument("--train-split", type=float, default=0.9,
                   help="Fraction of training data used for training.")
    d.add_argument("--use-augmentation", action=argparse.BooleanOptionalAction, default=True,
                   help="Enable RandAugment / MixUp augmentation.")
    d.add_argument("--randaugment-n", type=int, default=2,
                   help="RandAugment number of operations.")
    d.add_argument("--randaugment-m", type=int, default=7,
                   help="RandAugment magnitude.")
    d.add_argument("--mixup-alpha", type=float, default=0.25,
                   help="MixUp alpha (0 disables).")
    d.add_argument("--cutmix-alpha", type=float, default=0.25,
                   help="CutMix alpha (0 disables).")
    d.add_argument("--mean", type=float, nargs=3, default=None, metavar=("R", "G", "B"),
                   help="Override normalization mean. Defaults to dataset registry value.")
    d.add_argument("--std", type=float, nargs=3, default=None, metavar=("R", "G", "B"),
                   help="Override normalization std. Defaults to dataset registry value.")

    # ---------------------------------- Log ----------------------------------
    lg = parser.add_argument_group("Logging")
    lg.add_argument("--log-level", type=str, default="info",
                    choices=["debug", "info", "warning", "error"],
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
    lg.add_argument("--log-interval", type=int, default=50,
                    help="Log training metrics every N batches.")

    return parser


def parse_args(argv: Optional[list[str]] = None) -> Config:
    """
    Parse command-line arguments and return a structured `Config`.

    Steps
    -----
    1.  Parse `argv` with the full `ArgumentParser`.
    2.  Run cross-argument validation.
    3.  Normalize string values and resolve log level.
    4.  Assemble and return a `Config` instance.

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
    ns.scheduler = ns.scheduler.lower()
    ns.init_weights = ns.init_weights.lower()
    ns.cmt_variant = ns.cmt_variant.lower()

    return Config.from_namespace(ns)
