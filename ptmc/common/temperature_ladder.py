from __future__ import annotations
from typing import Literal
import numpy as np

LadderMethod = Literal["beta", "linear", "temp", "temperature"]

def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """
    Integrate y(x) using NumPy's trapezoid rule.
    This wrapper keeps compatibility with older NumPy versions where
    np.trapezoid may not exist yet.
    """
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:
        integrate = np.trapz
    return float(integrate(y, x))


def make_temperature_ladder(
    T_min: float,
    T_max: float,
    n_T: int,
    method: LadderMethod = "beta",
    dense_near_tc: bool = False,
    Tc: float = 1.0,
    tc_window: float = 0.05,
    tc_fraction: float = 0.50,
) -> np.ndarray:
    """
    Build a strictly increasing temperature ladder for parallel tempering.

    Parameters
    ----------
    T_min:
        Lowest temperature.
    T_max:
        Highest temperature.
    n_T:
        Number of temperatures / replicas.
    method:
        How to space the ladder.
        "beta":
            Uniform spacing in inverse temperature beta = 1 / T.
        "linear", "temp", "temperature":
            Uniform spacing directly in temperature T.
    dense_near_tc:
        If True, place more temperatures near Tc.
    Tc:
        Center of the dense temperature region.
    tc_window:
        Width of the dense temperature region.
    tc_fraction:
        Approximate fraction of ladder density assigned near Tc.
    Returns
    -------
    temps:
        A 1D float32 NumPy array of shape (n_T,), strictly increasing.
    """
    T_min = float(T_min)
    T_max = float(T_max)
    n_T = int(n_T)
    method = str(method).lower()
    if not np.isfinite(T_min) or not np.isfinite(T_max):
        raise ValueError("T_min and T_max must be finite.")
    if T_min <= 0.0 or T_max <= 0.0:
        raise ValueError("Temperatures must be positive.")
    if T_max <= T_min:
        raise ValueError("T_max must be larger than T_min.")
    if n_T < 2:
        raise ValueError("Parallel tempering requires at least two temperatures.")
    if method not in {"beta", "linear", "temp", "temperature"}:
        raise ValueError(
            f"Unsupported ladder method {method!r}. "
            "Use 'beta', 'linear', 'temp', or 'temperature'."
        )
    if not dense_near_tc:
        if method == "beta":
            betas = np.linspace(1.0 / T_min, 1.0 / T_max, n_T)
            temps = 1.0 / betas
        else:
            temps = np.linspace(T_min, T_max, n_T)
        return temps.astype(np.float32)
    if tc_window <= 0.0:
        raise ValueError("tc_window must be positive when dense_near_tc=True.")
    if not (0.0 < tc_fraction < 1.0):
        raise ValueError("tc_fraction must lie strictly between 0 and 1.")
    T_lo = max(T_min, float(Tc) - float(tc_window) / 2.0)
    T_hi = min(T_max, float(Tc) + float(tc_window) / 2.0)
    if T_hi <= T_lo:
        raise ValueError("Dense Tc window does not overlap the temperature range.")
    if method == "beta":
        coord_min = 1.0 / T_max
        coord_max = 1.0 / T_min
        def temp_to_coord(temp: float) -> float:
            return 1.0 / temp
        def coord_to_temp(coord: np.ndarray) -> np.ndarray:
            return 1.0 / coord
    else:
        coord_min = T_min
        coord_max = T_max
        def temp_to_coord(temp: float) -> float:
            return temp
        def coord_to_temp(coord: np.ndarray) -> np.ndarray:
            return coord
    coord_lo, coord_hi = sorted((temp_to_coord(T_lo), temp_to_coord(T_hi)))
    coord_span = coord_max - coord_min
    window_span = coord_hi - coord_lo
    if coord_span <= 0.0 or window_span <= 0.0:
        raise ValueError("Degenerate temperature ladder span.")
    n_grid = max(4097, 64 * n_T + 1)
    coord_grid = np.linspace(coord_min, coord_max, n_grid, dtype=np.float32)
    shoulder = max(
        window_span / 8.0,
        coord_span / max(8 * max(n_T - 1, 1), 128),
    )
    profile = 0.5 * (
        np.tanh((coord_grid - coord_lo) / shoulder)
        - np.tanh((coord_grid - coord_hi) / shoulder)
    )
    hard_window = (
        (coord_grid >= coord_lo) & (coord_grid <= coord_hi)
    ).astype(np.float32)
    base_total_mass = coord_span
    base_window_mass = window_span
    profile_total_mass = _trapezoid(profile, coord_grid)
    profile_window_mass = _trapezoid(profile * hard_window, coord_grid)
    def window_fraction(amplitude: float) -> float:
        numerator = base_window_mass + amplitude * profile_window_mass
        denominator = base_total_mass + amplitude * profile_total_mass
        return numerator / denominator
    base_fraction = base_window_mass / base_total_mass
    target_fraction = max(base_fraction, float(tc_fraction))
    amplitude = 0.0
    if target_fraction > base_fraction + 1.0e-12 and profile_total_mass > 0.0:
        amp_lo = 0.0
        amp_hi = 1.0
        while window_fraction(amp_hi) < target_fraction and amp_hi < 1.0e12:
            amp_hi *= 2.0
        if window_fraction(amp_hi) >= target_fraction:
            for _ in range(80):
                amp_mid = 0.5 * (amp_lo + amp_hi)
                if window_fraction(amp_mid) < target_fraction:
                    amp_lo = amp_mid
                else:
                    amp_hi = amp_mid
            amplitude = amp_hi
        else:
            amplitude = amp_hi
    density = 1.0 + amplitude * profile
    cdf = np.zeros_like(coord_grid)
    cdf[1:] = np.cumsum(
        0.5 * (density[1:] + density[:-1]) * np.diff(coord_grid)
    )
    total_mass = cdf[-1]
    if total_mass <= 0.0:
        raise RuntimeError("Temperature ladder density integration failed.")
    cdf /= total_mass
    coords = np.interp(
        np.linspace(0.0, 1.0, n_T),
        cdf,
        coord_grid,
    )
    temps = np.sort(coord_to_temp(coords).astype(np.float32))
    # Force exact endpoints after interpolation.
    temps[0] = np.float32(T_min)
    temps[-1] = np.float32(T_max)
    if np.any(np.diff(temps) <= 0.0):
        raise RuntimeError("Temperature ladder is not strictly increasing.")
    return temps


def temperature_ladder_diagnostics(temps: np.ndarray) -> dict[str, float]:
    """
    Summarize adjacent temperature spacing.

    This helps detect ladders with abrupt spacing jumps, which can hurt
    parallel tempering swap rates.
    """
    temps = np.asarray(temps, dtype=np.float32)

    if temps.ndim != 1:
        raise ValueError("temps must be a one-dimensional array.")
    dtemps = np.diff(temps)
    dtemps = dtemps[dtemps > 0.0]
    if dtemps.size == 0:
        return {
            "min_delta_T": 0.0,
            "max_delta_T": 0.0,
            "delta_T_gap_ratio": 0.0,
            "max_adjacent_delta_ratio": 0.0,
        }
    min_delta_T = float(np.min(dtemps))
    max_delta_T = float(np.max(dtemps))
    if dtemps.size > 1:
        local_ratios = np.maximum(
            dtemps[1:] / dtemps[:-1],
            dtemps[:-1] / dtemps[1:],
        )
        max_adjacent_delta_ratio = float(np.max(local_ratios))
    else:
        max_adjacent_delta_ratio = 1.0
    return {
        "min_delta_T": min_delta_T,
        "max_delta_T": max_delta_T,
        "delta_T_gap_ratio": (
            float(max_delta_T / min_delta_T)
            if min_delta_T > 0.0
            else float("inf")
        ),
        "max_adjacent_delta_ratio": max_adjacent_delta_ratio,
    }