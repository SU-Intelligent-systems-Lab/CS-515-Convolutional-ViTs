from models.cvt import CvT
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


__all__ = [
    "CvT",
    "build_cvt"
]
