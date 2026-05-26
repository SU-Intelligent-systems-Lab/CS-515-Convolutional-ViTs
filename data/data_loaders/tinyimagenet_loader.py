"""
TinyImageNet dataset and DataLoader factory.

Handles everything between raw files on disk and batches of tensors:

- `TinyImageNetDataset`: `torch.utils.data.Dataset` wrapping the official folder structure.
- `build_dataloaders`: returns train / val / test `DataLoader` instances configured from a `Config`.

Tiny ImageNet directory layout
-------------------------------
After unzipping `tiny-imagenet-200.zip`:

    tiny-imagenet-200/
    │-- train/
    │   │-- n01234567/      (nXXXXXXXX is a class ID)
    │   │   │-- images/*.JPEG
    │   │-- ... (200 class folders)
    │
    │-- val/
    │   │-- images/*.JPEG
    │   │-- val_annotations.txt     maps filename -> class ID
    │
    │-- test/
        │-- images/                 no labels,  skipped

The official test set has no labels, so the labeled val set is split 50/50 into a validation subset and a held-out
test subset.
"""
from pathlib import Path
from typing import Callable, Optional
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets.folder import default_loader
from data.transforms import get_transforms
from parameters import Config
from utils import get_logger, measure_time


logger = get_logger()


class TinyImageNetDataset(Dataset):
    """
    PyTorch Dataset for Tiny ImageNet-200.

    Supports the "train" and "val" splits. For the val split, class labels are parsed from `val_annotations.txt` and
    mapped to the same integer indices as the training split.

    Args:
        root: Path to the `tiny-imagenet-200` directory.
        split: "train" or "val".
        transform: Optional callable applied to each PIL image.
        class_to_idx: Pre-built {class_id: int} mapping. Pass the training dataset's mapping when constructing the val
                      dataset to guarantee consistent label assignment across splits.

    Attributes:
        class_to_idx: {class_id: int} mapping.
        idx_to_class: Inverse of class_to_idx.
        samples: List of (image_path, label_int) tuples.

    Raises:
        FileNotFoundError: If `root` or required subdirectories are missing.
        ValueError: If `split` is not "train" or "val".
    """
    SPLITS = ("train", "val")

    def __init__(self, root: str | Path, split: str = "train", transform: Optional[Callable] = None,
                 class_to_idx: Optional[dict[str, int]] = None) -> None:
        super().__init__()

        if split not in self.SPLITS:
            raise ValueError(f"split must be one of {self.SPLITS}, got '{split}'")

        self.root = Path(root)
        self.split = split
        self.transform = transform
        self._loader = default_loader

        if not self.root.exists():
            raise FileNotFoundError(
                f"Tiny ImageNet root not found: {self.root}\n "
                f"Download from http://cs231n.stanford.edu/tiny-imagenet-200.zip"
            )

        self.class_to_idx: dict[str, int] = class_to_idx if class_to_idx is not None else self._build_class_idx_map()

        self.idx_to_class: dict[int, str] = {v: k for k, v in self.class_to_idx.items()}

        self.class_to_readable_class: dict[str, str] = self._build_class_to_readable_class_map()

        self.idx_to_readable_class: dict[int, str] = {k: self.class_to_readable_class[v] for k, v in
                                                      self.idx_to_class.items()}

        self.readable_class_to_idx: dict[str, int] = {v: k for k, v in self.idx_to_readable_class.items()}

        self.samples: list[tuple[Path, int]] = (
            self._load_train_samples() if split == "train" else self._load_val_samples()
        )

        logger.info(f"TinyImageNet [{split}]: {len(self.samples)} samples, {len(self.class_to_idx)} classes")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        """
        Return (image_tensor, label) for the sample at specific index `idx`.

        Args:
            idx: Sample index in [0, len(self)).

        Returns:
            Tuple of (transformed_image, integer_label).
        """
        path, label = self.samples[idx]
        image = self._loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def _build_class_idx_map(self) -> dict[str, int]:
        """
        Build `{class_id: int}` from `wnids.txt` file   .

        Returns:
            Alphabetically sorted mapping of class IDs to integers 0–199.
        """
        wnids_file = self.root / "wnids.txt"
        if not wnids_file.exists():
            raise FileNotFoundError(f"wnids.txt not found under {self.root}")
        with open(wnids_file, "r") as f_wnids:
            wnids = [line.strip() for line in f_wnids]
        return {cls: i for i, cls in enumerate(sorted(wnids))}

    def _build_class_to_readable_class_map(self) -> dict[str, str]:
        """
        Build `{class_id: readable_label}` mapping from `wnids.txt` and `words.txt`.

        Returns:
            Mapping of class IDs to human-readable labels.
        """
        wnids_file = self.root / "wnids.txt"
        words_file = self.root / "words.txt"

        if not wnids_file.exists() or not words_file.exists():
            raise FileNotFoundError(f"wnids.txt or words.txt not found under {self.root}")

        # Build mapping from class ID to human-readable label
        class_to_readable: dict[str, str] = {}
        with open(wnids_file, "r") as f_wnids, open(words_file, "r") as f_words:
            wnids = [line.strip() for line in f_wnids]
            for line in f_words:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                class_id, readable_label = parts[0], parts[1]
                if class_id in wnids:
                    class_to_readable[class_id] = readable_label
        return class_to_readable

    def _load_train_samples(self) -> list[tuple[Path, int]]:
        """
        Collect all training images with their integer labels.

        Returns:
            List of (image_path, label_int) tuples.
        """
        samples: list[tuple[Path, int]] = []
        train_dir = self.root / "train"
        for class_id, label in self.class_to_idx.items():
            img_dir = train_dir / class_id / "images"
            if not img_dir.exists():
                continue
            for img_path in sorted(img_dir.iterdir()):
                if img_path.suffix.lower() in (".jpeg", ".jpg", ".png"):
                    samples.append((img_path, label))
        return samples

    def _load_val_samples(self) -> list[tuple[Path, int]]:
        """
        Parse `val_annotations.txt` and collect validation images.

        Returns:
            List of (image_path, label_int) tuples.
        """
        val_dir = self.root / "val"
        annotation_file = val_dir / "val_annotations.txt"
        img_dir = val_dir / "images"

        if not annotation_file.exists():
            raise FileNotFoundError(f"val_annotations.txt not found at {annotation_file}")

        samples: list[tuple[Path, int]] = []
        with open(annotation_file, "r") as fh:
            for line in fh:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                filename, class_id = parts[0], parts[1]
                if class_id not in self.class_to_idx:
                    continue
                img_path = img_dir / filename
                if img_path.exists():
                    samples.append((img_path, self.class_to_idx[class_id]))
        return samples


# ------------- DataLoader factory -------------

@measure_time
def build_dataloaders(cfg: Config) -> dict[str, DataLoader]:
    """
    Build train, val, and test `DataLoader` instances for Tiny ImageNet.

    The official Tiny ImageNet test set has no labels, so the labeled val split (10,000 images) is divided 50/50 into
    a validation subset and a held-out test subset using a fixed seed for reproducibility.

    Any Augmentation is not applied here: they operate on full batches and are applied in the training loop after
    calling this function.

    Args:
        cfg: `Config` instance with all data hyperparameters.

    Returns:
        Dictionary with keys "train", "val", "test" mapping to `DataLoader` instances.

    Raises:
        FileNotFoundError: Propagated from `TinyImageNetDataset` when the dataset root is missing.

    Example:
        >>> loaders = build_dataloaders(Config())
        >>> images, labels = next(iter(loaders["train"]))
        >>> images.shape
        torch.Size([128, 3, 64, 64])
    """
    def make_generator():
        """Generator seed ensures val/test indices are identical across the two dataset objects (raw and
        val-transformed)"""
        return torch.Generator().manual_seed(42)

    train_tf = get_transforms(cfg.data, train=True)
    eval_tf = get_transforms(cfg.data, train=False)

    train_dataset = TinyImageNetDataset(root=cfg.data.data_dir, split="train", transform=train_tf)

    # In validation dataset, we pass training mapping for consistent label integers across splits
    val_dataset = TinyImageNetDataset(root=cfg.data.data_dir, split="val", transform=eval_tf,
                                      class_to_idx=train_dataset.class_to_idx)

    # Splitting Val into (val + test)
    val_size = len(val_dataset) // 2
    test_size = len(val_dataset) - val_size
    val_subset, test_subset = random_split(val_dataset, [val_size, test_size], generator=make_generator())

    logger.info(f"Dataset sizes: train: {len(train_dataset)} | val: {len(val_subset)} | test: {len(test_subset)}")

    # ---------- Shared DataLoader kwargs ----------
    _common_dataloader_kwargs: dict = dict(
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        prefetch_factor=cfg.data.prefetch_factor if cfg.data.num_workers > 0 else None,
        persistent_workers=cfg.data.num_workers > 0,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        drop_last=True,   # uniform batch size; important for BatchNorm
        **_common_dataloader_kwargs,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        drop_last=False,
        **_common_dataloader_kwargs,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        drop_last=False,
        **_common_dataloader_kwargs,
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}
