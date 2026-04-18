import numpy as np
import importlib
from typing import TYPE_CHECKING

# Provide a backward-compat alias only if missing, and map to numpy.bool_
# Using Python's built-in bool here can break libraries that expect a NumPy dtype
if not hasattr(np, "bool"):
    np.bool = np.bool_

# Lazily expose common submodules to avoid heavy import-time costs.
# Submodules will be loaded on first attribute access or when imported directly
# (e.g., `from meta_material import utils` or `import meta_material.utils`).
__all__ = [
    "utils",
    "ffmpeg",
    "warp",
    "data",
    "material",
    "sim",
]


def __getattr__(name: str):
    """Lazy attribute access for submodules with clear, fail-fast errors.

    This keeps `import meta_material` fast while still supporting
    `meta_material.utils` style access. If a submodule fails to import due to a
    missing optional dependency or other import-time error, we raise an
    informative ImportError rather than silently failing.
    """
    if name in __all__:
        try:
            module = importlib.import_module(f"{__name__}.{name}")
        except ImportError as exc:
            raise ImportError(
                (
                    f"Failed to import submodule '{name}' from package '{__name__}'. "
                    f"Original error: {exc.__class__.__name__}: {exc}. "
                    "This likely indicates a missing optional dependency or an "
                    "import-time error within that submodule. Install the required "
                    "dependencies for this feature and try again."
                )
            ) from exc
        # Cache on the package for subsequent attribute lookups
        globals()[name] = module
        return module
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__():
    # Help tools like dir() and IDEs discover lazy attributes
    return sorted(list(globals().keys()) + __all__)


if TYPE_CHECKING:
    # These are only for static type checkers; they do not execute at runtime.
    from . import utils as utils  # noqa: F401
    from . import ffmpeg as ffmpeg  # noqa: F401
    from . import warp as warp  # noqa: F401
    from . import data as data  # noqa: F401
    from . import material as material  # noqa: F401
    from . import sim as sim  # noqa: F401
