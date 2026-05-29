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

def compute_helicity_observables(
    *,
    temps: Any,
    L: int,
    helicity_Kx_block_means: Any,
    helicity_Ix_block_means: Any,
    helicity_Ix2_block_means: Any,
    helicity_Ky_block_means: Any,
    helicity_Iy_block_means: Any,
    helicity_Iy2_block_means: Any,
) -> dict[str, np.ndarray]:
    """
    Compute helicity modulus observables from block means.

    Parameters
    ----------
    temps:
        Temperature array of shape (R,).
    L:
        Linear system size. The number of sites is N = L * L.
    helicity_Kx_block_means:
        Block means of K_x.
    helicity_Ix_block_means:
        Block means of I_x.
    helicity_Ix2_block_means:
        Block means of I_x^2.
    helicity_Ky_block_means:
        Block means of K_y.
    helicity_Iy_block_means:
        Block means of I_y.
    helicity_Iy2_block_means:
        Block means of I_y^2.

    Returns
    -------
    Dictionary containing:
        helicity_Kx, helicity_Kx_err
        helicity_Ix, helicity_Ix_err
        helicity_Ix2, helicity_Ix2_err
        helicity_Ky, helicity_Ky_err
        helicity_Iy, helicity_Iy_err
        helicity_Iy2, helicity_Iy2_err
        helicity_Yx, helicity_Yx_err
        helicity_Yy, helicity_Yy_err
        Y, Y_err
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
    Kx_blocks = _as_block_matrix(
        helicity_Kx_block_means,
        name="helicity_Kx_block_means",
        n_temps=n_temps,
    )
    Ix_blocks = _as_block_matrix(
        helicity_Ix_block_means,
        name="helicity_Ix_block_means",
        n_temps=n_temps,
    )
    Ix2_blocks = _as_block_matrix(
        helicity_Ix2_block_means,
        name="helicity_Ix2_block_means",
        n_temps=n_temps,
    )
    Ky_blocks = _as_block_matrix(
        helicity_Ky_block_means,
        name="helicity_Ky_block_means",
        n_temps=n_temps,
    )
    Iy_blocks = _as_block_matrix(
        helicity_Iy_block_means,
        name="helicity_Iy_block_means",
        n_temps=n_temps,
    )
    Iy2_blocks = _as_block_matrix(
        helicity_Iy2_block_means,
        name="helicity_Iy2_block_means",
        n_temps=n_temps,
    )
    helicity_Kx = np.full(n_temps, np.nan)
    helicity_Kx_err = np.full(n_temps, np.nan)
    helicity_Ix = np.full(n_temps, np.nan)
    helicity_Ix_err = np.full(n_temps, np.nan)
    helicity_Ix2 = np.full(n_temps, np.nan)
    helicity_Ix2_err = np.full(n_temps, np.nan)
    helicity_Ky = np.full(n_temps, np.nan)
    helicity_Ky_err = np.full(n_temps, np.nan)
    helicity_Iy = np.full(n_temps, np.nan)
    helicity_Iy_err = np.full(n_temps, np.nan)
    helicity_Iy2 = np.full(n_temps, np.nan)
    helicity_Iy2_err = np.full(n_temps, np.nan)
    helicity_Yx = np.full(n_temps, np.nan)
    helicity_Yx_err = np.full(n_temps, np.nan)
    helicity_Yy = np.full(n_temps, np.nan)
    helicity_Yy_err = np.full(n_temps, np.nan)
    Y = np.full(n_temps, np.nan)
    Y_err = np.full(n_temps, np.nan)
    helicity_Ix_mean_over_rms = np.full(n_temps, np.nan)
    helicity_Iy_mean_over_rms = np.full(n_temps, np.nan)
    for r in range(n_temps):
        beta = float(betas[r])
        helicity_Kx[r], helicity_Kx_err[r] = jackknife_from_block_means(
            Kx_blocks[r]
        )
        helicity_Ix[r], helicity_Ix_err[r] = jackknife_from_block_means(
            Ix_blocks[r]
        )
        helicity_Ix2[r], helicity_Ix2_err[r] = jackknife_from_block_means(
            Ix2_blocks[r]
        )

        helicity_Ky[r], helicity_Ky_err[r] = jackknife_from_block_means(
            Ky_blocks[r]
        )
        helicity_Iy[r], helicity_Iy_err[r] = jackknife_from_block_means(
            Iy_blocks[r]
        )
        helicity_Iy2[r], helicity_Iy2_err[r] = jackknife_from_block_means(
            Iy2_blocks[r]
        )
        Ix_mean = float(np.mean(Ix_blocks[r]))
        Ix2_mean = float(np.mean(Ix2_blocks[r]))
        Iy_mean = float(np.mean(Iy_blocks[r]))
        Iy2_mean = float(np.mean(Iy2_blocks[r]))
        Ix_rms = np.sqrt(max(Ix2_mean, 0.0))
        Iy_rms = np.sqrt(max(Iy2_mean, 0.0))
        helicity_Ix_mean_over_rms[r] = (
            abs(Ix_mean) / Ix_rms if Ix_rms > 0.0 else np.nan
        )
        helicity_Iy_mean_over_rms[r] = (
            abs(Iy_mean) / Iy_rms if Iy_rms > 0.0 else np.nan
        )
        def estimate_Yx(means: dict[str, np.ndarray]) -> float:
            Kx = float(means["Kx"][0])
            Ix = float(means["Ix"][0])
            Ix2 = float(means["Ix2"][0])
            return (Kx - beta * (Ix2 - Ix * Ix)) / N
        helicity_Yx[r], helicity_Yx_err[r] = jackknife_apply(
            {
                "Kx": Kx_blocks[r],
                "Ix": Ix_blocks[r],
                "Ix2": Ix2_blocks[r],
            },
            estimate_Yx,
        )
        def estimate_Yy(means: dict[str, np.ndarray]) -> float:
            Ky = float(means["Ky"][0])
            Iy = float(means["Iy"][0])
            Iy2 = float(means["Iy2"][0])
            return (Ky - beta * (Iy2 - Iy * Iy)) / N
        helicity_Yy[r], helicity_Yy_err[r] = jackknife_apply(
            {
                "Ky": Ky_blocks[r],
                "Iy": Iy_blocks[r],
                "Iy2": Iy2_blocks[r],
            },
            estimate_Yy,
        )
        def estimate_Y(means: dict[str, np.ndarray]) -> float:
            Kx = float(means["Kx"][0])
            Ix = float(means["Ix"][0])
            Ix2 = float(means["Ix2"][0])
            Ky = float(means["Ky"][0])
            Iy = float(means["Iy"][0])
            Iy2 = float(means["Iy2"][0])
            Yx = (Kx - beta * (Ix2 - Ix * Ix)) / N
            Yy = (Ky - beta * (Iy2 - Iy * Iy)) / N
            return 0.5 * (Yx + Yy)
        Y[r], Y_err[r] = jackknife_apply(
            {
                "Kx": Kx_blocks[r],
                "Ix": Ix_blocks[r],
                "Ix2": Ix2_blocks[r],
                "Ky": Ky_blocks[r],
                "Iy": Iy_blocks[r],
                "Iy2": Iy2_blocks[r],
            },
            estimate_Y,
        )
    return {
        "helicity_Kx": helicity_Kx,
        "helicity_Kx_err": helicity_Kx_err,
        "helicity_Ix": helicity_Ix,
        "helicity_Ix_err": helicity_Ix_err,
        "helicity_Ix2": helicity_Ix2,
        "helicity_Ix2_err": helicity_Ix2_err,
        "helicity_Ky": helicity_Ky,
        "helicity_Ky_err": helicity_Ky_err,
        "helicity_Iy": helicity_Iy,
        "helicity_Iy_err": helicity_Iy_err,
        "helicity_Iy2": helicity_Iy2,
        "helicity_Iy2_err": helicity_Iy2_err,
        "helicity_Yx": helicity_Yx,
        "helicity_Yx_err": helicity_Yx_err,
        "helicity_Yy": helicity_Yy,
        "helicity_Yy_err": helicity_Yy_err,
        "Y": Y,
        "Y_err": Y_err,
        "helicity_Ix_mean_over_rms": helicity_Ix_mean_over_rms,
        "helicity_Iy_mean_over_rms": helicity_Iy_mean_over_rms,
    }