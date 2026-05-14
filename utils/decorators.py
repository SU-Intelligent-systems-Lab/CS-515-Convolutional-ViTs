import functools
import logging
import time
from typing import Any, Callable, TypeVar


F = TypeVar("F", bound=Callable[..., Any])

logger = logging.getLogger("convvits")


def measure_time(func: F) -> F:
    """Decorator that logs the wall-clock execution time of a function.

    The elapsed time is also injected into the function's return value when the return value is a ``dict``
    (e.g. training/evaluation result dicts), under the key ``"elapsed_sec"``.

    Args:
        func: The function to wrap.

    Returns:
        Wrapped function with timing instrumentation.

    Example:
        >>> @measure_time
        ... def train_one_epoch(model, loader):
        ...     ...
        ...     return {"loss": 0.42}
        >>> result = train_one_epoch(model, loader)
        >>> print(result["elapsed_sec"])   # seconds as float
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        duration = end_time - start_time
        logger.info(f"'{func.__name__}' in '{func.__module__}' executed in {duration:.4f}s")

        # Inject timing into dict results so callers can log it easily.
        if isinstance(result, dict):
            result["elapsed_sec"] = duration

        return result

    return wrapper
