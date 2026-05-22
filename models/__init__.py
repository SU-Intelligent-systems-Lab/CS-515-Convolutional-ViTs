from typing import Any
from torch import nn
from models.cvt import CvT
from models.cmt import CMT
from parameters import Config


def build_cvt(config: Config) -> CvT:
    """
    Construct a `CvT` model from a `Config`.

    A thin factory wrapper that makes `main.py` read cleanly:
    `model = build_cvt(cfg)` rather than `CvT(cfg)`.

    Args:
        config: Populated `Config` dataclass.

    Returns:
        Initialised `CvT` instance.
    """
    return CvT(config)

def build_cmt(config: Config) -> CMT:
    """
    Construct a `CMT` model from a `Config`.

    A thin factory wrapper that makes `main.py` read cleanly:
    `model = build_cmt(cfg)` rather than `CMT(cfg)`.

    Args:
        config: Populated `Config` dataclass.

    Returns:
        Initialised `CMT` instance.
    """
    return CMT(config)


# Model Registry: Maps model name strings -> factory functions. Each factory receives the full `Config` and
#                 returns an `nn.Module`.
_MODEL_REGISTRY: dict[str, Any] = {
    "cvt": build_cvt,
    "cmt": build_cmt,
}


def build_model(cfg: Config) -> nn.Module:
    """
    Instantiate the model specified by `cfg.model`. Dispatches on the `--model-name` CLI argument. Extensible when
    adding new architectures to `_MODEL_REGISTRY` without changing any other file.

    Args:
        cfg: Full `Config` instance. The model factory receives `cfg.model` (a `ModelConfig` dataclass).

    Returns:
        Randomly initialized `nn.Module`.

    Raises:
        ValueError: If the model name is not in `_MODEL_REGISTRY`.
    """
    model_name = getattr(cfg.model, "model_name", "cvt")

    if model_name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(_MODEL_REGISTRY.keys())}")

    factory = _MODEL_REGISTRY[model_name]
    model = factory(cfg)
    return model


__all__ = [
    "build_model",
    "CvT",
    "build_cvt",
    "CMT",
    "build_cmt"
]
