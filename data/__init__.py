from typing import Any
from data.data_loaders.tinyimagenet_loader import build_dataloaders as build_tinyimagenet_dataloaders
from parameters import Config


_DATA_LOADERS: dict[str, Any] = {
    "tiny-imagenet-200": build_tinyimagenet_dataloaders,
}


def build_dataloaders(cfg: Config):
    dataset = getattr(cfg.data, "dataset", "tiny-imagenet-200")

    if dataset not in _DATA_LOADERS:
        raise ValueError(f"Unknown dataset '{dataset}'. Available: {list(_DATA_LOADERS.keys())}")

    data_loader_factory = _DATA_LOADERS[dataset]
    return data_loader_factory(cfg)


__all__ = [
    "build_tinyimagenet_dataloaders",
    "build_dataloaders"
]
