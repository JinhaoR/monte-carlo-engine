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

def compute_z2_observables(
    *,
    temps: ArrayLike,
    L: int,
    order_abs_block_means: ArrayLike,
    order2_block_means: ArrayLike,
    order4_block_means: ArrayLike,
    order_parameter_per_site: bool = False,
) -> dict[str, np.ndarray]:
    """
    Compute Z2 order parameter observables.

    Parameters
    ----------
    temps:
        Temperature array of shape (R,).
    L:
        Linear system size. The number of sites is N = L * L.
    order_abs_block_means:
        Block means of |M| or |m|.
    order2_block_means:
        Block means of M^2 or m^2.
    order4_block_means:
        Block means of M^4 or m^4.
    order_parameter_per_site:
        False:
            blocks store total order parameter M.
        True:
            blocks store order parameter density m = M / N.
    Returns
    -------
    Dictionary containing:
        m_abs, m_abs_err
        chi, chi_err
        U4, U4_err
        binder_ratio, binder_ratio_err
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
    Mabs_blocks = _as_block_matrix(
        order_abs_block_means,
        name="order_abs_block_means",
        n_temps=n_temps,
    )
    M2_blocks = _as_block_matrix(
        order2_block_means,
        name="order2_block_means",
        n_temps=n_temps,
    )
    M4_blocks = _as_block_matrix(
        order4_block_means,
        name="order4_block_means",
        n_temps=n_temps,
    )
    m_abs = np.full(n_temps, np.nan)
    m_abs_err = np.full(n_temps, np.nan)
    chi = np.full(n_temps, np.nan)
    chi_err = np.full(n_temps, np.nan)
    U4 = np.full(n_temps, np.nan)
    U4_err = np.full(n_temps, np.nan)
    binder_ratio = np.full(n_temps, np.nan)
    binder_ratio_err = np.full(n_temps, np.nan)
    order_scale = 1.0 if order_parameter_per_site else 1.0 / N
    for r in range(n_temps):
        beta = float(betas[r])
        m_abs[r], m_abs_err[r] = jackknife_from_block_means(
            Mabs_blocks[r] * order_scale
        )
        if order_parameter_per_site:
            # Blocks contain m, |m|, m^2, m^4.
            #
            # chi = beta * N * (<m^2> - <|m|>^2)
            chi_prefactor = beta * N
        else:
            # Blocks contain M, |M|, M^2, M^4.
            #
            # chi = beta / N * (<M^2> - <|M|>^2)
            chi_prefactor = beta / N
        def estimate_susceptibility(means: dict[str, np.ndarray]) -> float:
            Mabs_mean = float(means["Mabs"][0])
            M2_mean = float(means["M2"][0])
            return chi_prefactor * (M2_mean - Mabs_mean * Mabs_mean)
        chi[r], chi_err[r] = jackknife_apply(
            {
                "Mabs": Mabs_blocks[r],
                "M2": M2_blocks[r],
            },
            estimate_susceptibility,
        )
        def estimate_binder(means: dict[str, np.ndarray]) -> float:
            M2_mean = float(means["M2"][0])
            M4_mean = float(means["M4"][0])
            if M2_mean <= 0.0:
                return np.nan
            return 1.0 - M4_mean / (3.0 * M2_mean * M2_mean)
        U4[r], U4_err[r] = jackknife_apply(
            {
                "M2": M2_blocks[r],
                "M4": M4_blocks[r],
            },
            estimate_binder,
        )
        def estimate_binder_ratio(means: dict[str, np.ndarray]) -> float:
            M2_mean = float(means["M2"][0])
            M4_mean = float(means["M4"][0])
            if M2_mean <= 0.0:
                return np.nan
            return M4_mean / (3.0 * M2_mean * M2_mean)
        binder_ratio[r], binder_ratio_err[r] = jackknife_apply(
            {
                "M2": M2_blocks[r],
                "M4": M4_blocks[r],
            },
            estimate_binder_ratio,
        )
    return {
        "m_abs": m_abs,
        "m_abs_err": m_abs_err,
        "chi": chi,
        "chi_err": chi_err,
        "U4": U4,
        "U4_err": U4_err,
        "binder_ratio": binder_ratio,
        "binder_ratio_err": binder_ratio_err,
    }
