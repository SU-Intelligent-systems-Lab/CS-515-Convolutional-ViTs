from cvt import CvT
from parameters import ModelConfig


def build_cvt(config: ModelConfig) -> CvT:
    """
    Construct a `CvT` model from a `ModelConfig`.

    A thin factory wrapper that makes `main.py` read cleanly:
    `model = build_cvt(cfg.model)` rather than `CvT(cfg.model)`.

    Args:
        config: Populated `ModelConfig` dataclass.

    Returns:
        Initialised `CvT` instance.
    """
    return CvT(config)


__all__ = [
    "CvT",
    "build_cvt"
]
