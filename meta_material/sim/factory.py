from __future__ import annotations

from typing import Literal, Optional

import numpy as np

from .spring_mass import ImplicitBatchSim


SimBackend = Literal["spring"]


def create_simulator(
    *,
    backend: SimBackend,
    x: np.ndarray,
    v: np.ndarray,
    grippers: Optional[np.ndarray],
    points_per_env: int,
    batch_size: int,
    **kwargs,
):
    if backend == "spring":
        return ImplicitBatchSim(
            x=x,
            v=v,
            grippers=grippers,
            points_per_env=points_per_env,
            batch_size=batch_size,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unknown simulator backend: {backend}. Only 'spring' is supported."
        )
