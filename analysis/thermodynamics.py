from __future__ import annotations
from typing import Any
import numpy as np

from analysis.statistics import (
    jackknife_apply,
    jackknife_from_block_means,
)

ArrayLike = Any

def _as_block_matrix(
    values: ArrayLike,
    *,
    name: str,
    n_temps: int,
) -> np.ndarray:
    """
    Convert values to a block matrix of shape (n_temps, n_blocks).
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(
            f"{name} must be a 2D array with shape "
            f"(n_temps, n_blocks), got shape {arr.shape}."
        )
    if arr.shape[0] != n_temps:
        raise ValueError(
            f"{name} has wrong number of temperature rows. "
            f"Expected {n_temps}, got {arr.shape[0]}."
        )
    if arr.shape[1] <= 0:
        raise ValueError(f"{name} must contain at least one block.")
    return arr

def compute_thermodynamics(
    *,
    temps: ArrayLike,
    L: int,
    energy_block_means: ArrayLike,
    energy2_block_means: ArrayLike,
    energy_per_site: bool = False,
) -> dict[str, np.ndarray]:
    """
    Compute energy per site and specific heat per site.

    Parameters
    ----------
    temps:
        Temperature array of shape (R,).
    L:
        Linear system size. The number of sites is N = L * L.
    energy_block_means:
        Block means of either total energy E or energy density e = E / N.
    energy2_block_means:
        Block means of either E^2 or e^2.
    energy_per_site:
        False:
            energy_block_means stores E
            energy2_block_means stores E^2
        True:
            energy_block_means stores e = E / N
            energy2_block_means stores e^2
    Returns
    -------
    Dictionary containing:
        e, e_err
        C, C_err
    """
    temps = np.asarray(temps, dtype=np.float64)
    if temps.ndim != 1:
        raise ValueError("temps must be a one-dimensional array.")
    if np.any(~np.isfinite(temps)) or np.any(temps <= 0.0):
        raise ValueError("temps must contain finite positive temperatures.")
    L = int(L)
    if L <= 0:
        raise ValueError("L must be positive.")
    n_temps = temps.size
    N = L * L
    betas = 1.0 / temps
    E_blocks = _as_block_matrix(
        energy_block_means,
        name="energy_block_means",
        n_temps=n_temps,
    )
    E2_blocks = _as_block_matrix(
        energy2_block_means,
        name="energy2_block_means",
        n_temps=n_temps,
    )
    e = np.full(n_temps, np.nan)
    e_err = np.full(n_temps, np.nan)
    C = np.full(n_temps, np.nan)
    C_err = np.full(n_temps, np.nan)
    energy_scale = 1.0 if energy_per_site else 1.0 / N
    for r in range(n_temps):
        beta = float(betas[r])
        e[r], e_err[r] = jackknife_from_block_means(
            E_blocks[r] * energy_scale
        )
        if energy_per_site:
            # Blocks contain e and e^2.
            #
            # C per site = beta^2 * N * (<e^2> - <e>^2)
            C_prefactor = beta * beta * N
        else:
            # Blocks contain total E and E^2.
            #
            # C per site = beta^2 / N * (<E^2> - <E>^2)
            C_prefactor = beta * beta / N
        def estimate_specific_heat(means: dict[str, np.ndarray]) -> float:
            E_mean = float(means["E"][0])
            E2_mean = float(means["E2"][0])
            return C_prefactor * (E2_mean - E_mean * E_mean)
        C[r], C_err[r] = jackknife_apply(
            {
                "E": E_blocks[r],
                "E2": E2_blocks[r],
            },
            estimate_specific_heat,
        )
    return {
        "e": e,
        "e_err": e_err,
        "C": C,
        "C_err": C_err,
    }