from typing import Any
from data.data_loaders.tinyimagenet_loader import TinyImageNetDataset, build_dataloaders as build_tinyimagenet_dataloaders
from parameters import Config


_DATA_LOADERS: dict[str, Any] = {
    "tiny-imagenet-200": build_tinyimagenet_dataloaders,
}

_DATA_LOADER_CLASSES: dict[str, Any] = {
    "tiny-imagenet-200": TinyImageNetDataset,
}


def build_dataloaders(cfg: Config):
    dataset = getattr(cfg.data, "dataset", "tiny-imagenet-200")

    if dataset not in _DATA_LOADERS:
        raise ValueError(f"Unknown dataset '{dataset}'. Available: {list(_DATA_LOADERS.keys())}")

    data_loader_factory = _DATA_LOADERS[dataset]
    return data_loader_factory(cfg)


def get_class_name_index_map(cfg: Config):
    dataset = getattr(cfg.data, "dataset", "tiny-imagenet-200")
    if dataset not in _DATA_LOADER_CLASSES:
        raise ValueError(f"Unknown dataset '{dataset}'. Available: {list(_DATA_LOADER_CLASSES.keys())}")

    dataset = _DATA_LOADER_CLASSES[dataset](root=cfg.data.data_dir)
    return dataset.idx_to_readable_class


__all__ = [
    "build_tinyimagenet_dataloaders",
    "build_dataloaders",
    "get_class_name_index_map",
]
