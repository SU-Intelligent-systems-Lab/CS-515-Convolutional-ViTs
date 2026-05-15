"""
Dataset-agnostic image transform pipelines.

Provides a single `get_transforms(cfg, train)` entry-point with:
-   Dataset dispatch: branching on `cfg.dataset` so each dataset can have its own augmentation strategy. Adding a new
    dataset only requires a new `elif` branch.
-   `DataConfig`-typed config: no raw dicts; all fields are typed and validated at parse time.

The function is intentionally dataset-agnostic: It reads only `image_size`, `mean`, `std`, and augmentation
flags from the config, so it works unchanged for any dataset registered in `DataConfig`.
"""
from torchvision import transforms
from parameters import DataConfig


def get_transforms(cfg: DataConfig, train: bool = True) -> transforms.Compose:
    """
    Return the appropriate transform pipeline for the configured dataset.

    A single function that keeps the training/eval split logic in one place. Dataset-specific operations are handled
    via an internal dispatch so adding a new dataset only requires a new `elif` block below, nothing else.

    Args:
        cfg: `DataConfig` instance carrying dataset name, image size, augmentation flags, and normalization statistics.
        train: If `True`, return the stochastic training pipeline. If `False`, return the deterministic eval pipeline
            (no random operations: guarantees reproducible val / test metrics).

    Returns:
        `transforms.Compose` pipeline ready to be passed to a `Dataset` or `DataLoader`.

    Raises:
        ValueError: If `cfg.dataset` is not a recognized dataset key.

    Example:
        >>> cfg = DataConfig()
        >>> train_tf = get_transforms(cfg, train=True)
        >>> eval_tf  = get_transforms(cfg, train=False)
    """
    if cfg.dataset == "tiny-imagenet-200":
        return _tiny_imagenet_transforms(cfg, train)
    else:
        raise ValueError(
            f"No transform pipeline defined for dataset '{cfg.dataset}'. Add an elif branch in data/transforms.py."
        )


# ----------------------------------------
# Per-dataset pipelines
# ----------------------------------------

def _tiny_imagenet_transforms(cfg: DataConfig, train: bool,) -> transforms.Compose:
    """
    Build the Tiny ImageNet transform pipeline.

    Training pipeline
    -----------------
    1. `RandomResizedCrop(image_size, scale=(0.67, 1.0))`: Provides scale and position diversity. Critical for CvT 
        because unlike a pure CNN, convolutional token embedding gives only local positional encoding, it does not have
        the full translation invariance that pooling gives CNNs. The model must  learn scale-invariant features
        explicitly. `RandomResizedCrop` forces the model to classify objects regardless of their position and scale,
        compensating for this. Scale floor of 0.67 retains at least 67% of the original 64x64 image which is aggressive
        enough to be useful, but conservative enough not to destroy 64x64 images. For comparison,
        `RandomCrop(32, padding=4)` used on CIFAR-10 only shifts the image but it provides no scale diversity. CvT
        benefits more from scale than from shift, which is why `RandomResizedCrop` is the right choice here.
    2. `RandomHorizontalFlip()`: 50 % chance horizontal mirror.
    3. `RandAugment(n, m)`: samples n random operations from a pool of photometric / geometric transforms at
        magnitude m. Only applied when `cfg.use_augmentation` is True. Dramatically increases effective dataset size.
    4. `ToTensor()`: PIL -> float32 in [0, 1].
    5. `Normalize(mean, std)`: per-channel standardization using Tiny ImageNet pre-computed statistics stored in `cfg`.

    Eval pipeline (val / test)
    --------------------------
    1. `ToTensor()`
    2. `Normalize(mean, std)`: same statistics as training.

    Tiny ImageNet images are already exactly 64x64, so no resize or crop is needed.
    No random operations in the eval pipeline: reproducible metrics regardless of run order or random seed.
    Training randomly crops then scales back to 64x64, so evaluation should see the unmodified full 64x64 image.

    Args:
        cfg: `DataConfig` carrying augmentation flags and image stats.
        train: `True` -> stochastic training pipeline.
               `False` -> deterministic eval pipeline.

    Returns:
        `transforms.Compose` pipeline.
    """
    mean = list(cfg.mean)
    std = list(cfg.std)

    transform_list: list = []

    if train:
        transform_list.append([
            transforms.RandomResizedCrop(
                cfg.image_size,
                scale=(0.67, 1.0),
                ratio=(3 / 4, 4 / 3),
            ),
            transforms.RandomHorizontalFlip(),
        ])

        if cfg.use_augmentation:
            transform_list.append(
                transforms.RandAugment(
                    num_ops=cfg.randaugment_n,
                    magnitude=cfg.randaugment_m,
                )
            )
        transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    else:
        # Fixed-size dataset (64x64), no spatial transform needed.
        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]

    return transforms.Compose(transform_list)
