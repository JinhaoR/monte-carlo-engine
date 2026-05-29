from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable
import numpy as np

ArrayLike = Any

@dataclass(frozen=True)
class AutocorrelationResult:
    """
    Result from an integrated autocorrelation time estimate.
    """
    tau_int: float
    window: int
    n: int
    n_eff: float
    rho1: float
    mean: float
    naive_error: float
    autocorr_error: float


def finite_1d(samples: ArrayLike) -> np.ndarray:
    """
    Convert input to a finite 1D float64 array.
    """
    arr = np.asarray(samples, dtype=np.float64).ravel()
    return arr[np.isfinite(arr)]


def block_means(samples: ArrayLike, n_blocks: int) -> np.ndarray:
    """
    Split a 1D sample series into contiguous blocks and return block means.
    """
    x = finite_1d(samples)
    n = x.size
    if n == 0:
        return np.empty(0, dtype=np.float64)
    n_blocks = int(n_blocks)
    if n_blocks <= 0:
        raise ValueError("n_blocks must be positive.")
    n_blocks = min(n_blocks, n)
    block_size = n // n_blocks
    if block_size <= 0:
        return np.empty(0, dtype=np.float64)
    n_use = n_blocks * block_size
    blocks = x[:n_use].reshape(n_blocks, block_size)
    return np.mean(blocks, axis=1)


def jackknife_from_block_means(block_values: ArrayLike) -> tuple[float, float]:
    """
    Compute jackknife mean and error from precomputed block values.
    """
    blocks = finite_1d(block_values)
    n_blocks = blocks.size
    if n_blocks == 0:
        return np.nan, np.nan
    if n_blocks == 1:
        return float(blocks[0]), 0.0
    total = np.sum(blocks)
    jk = np.empty(n_blocks, dtype=np.float64)
    for i in range(n_blocks):
        jk[i] = (total - blocks[i]) / (n_blocks - 1)
    mean = float(np.mean(jk))
    var = (n_blocks - 1) / n_blocks * np.sum((jk - mean) ** 2)
    return mean, float(np.sqrt(var))


def jackknife_blocks(
    samples: ArrayLike,
    n_blocks: int = 20,
) -> tuple[float, float]:
    """
    Compute block jackknife mean and error from raw samples.
    """
    x = finite_1d(samples)
    n = x.size
    if n == 0:
        return np.nan, np.nan
    if n == 1:
        return float(x[0]), 0.0
    n_blocks = min(int(n_blocks), n // 2)
    if n_blocks < 2:
        mean = float(np.mean(x))
        err = float(np.std(x, ddof=1) / np.sqrt(n))
        return mean, err
    blocks = block_means(x, n_blocks)
    return jackknife_from_block_means(blocks)


def jackknife_apply(
    block_arrays: dict[str, ArrayLike],
    estimator: Callable[[dict[str, np.ndarray]], float],
) -> tuple[float, float]:
    """
    Generic jackknife for nonlinear observables.
    """
    arrays = {
        key: finite_1d(value)
        for key, value in block_arrays.items()
    }
    if not arrays:
        return np.nan, np.nan
    n_blocks_set = {arr.size for arr in arrays.values()}
    if len(n_blocks_set) != 1:
        raise ValueError("All block arrays must have the same number of blocks.")
    n_blocks = n_blocks_set.pop()
    if n_blocks == 0:
        return np.nan, np.nan
    if n_blocks == 1:
        means = {key: arr[0:1] for key, arr in arrays.items()}
        return float(estimator(means)), 0.0
    sums = {
        key: np.sum(arr)
        for key, arr in arrays.items()
    }
    jk = np.empty(n_blocks, dtype=np.float64)
    for i in range(n_blocks):
        leave_one_out_means = {
            key: np.asarray([(sums[key] - arr[i]) / (n_blocks - 1)])
            for key, arr in arrays.items()
        }
        jk[i] = estimator(leave_one_out_means)
    mean = float(np.mean(jk))
    var = (n_blocks - 1) / n_blocks * np.sum((jk - mean) ** 2)
    return mean, float(np.sqrt(var))


def autocorrelation_function(
    samples: ArrayLike,
    max_lag: int | None = None,
) -> np.ndarray:
    """
    Estimate the normalized autocorrelation function rho(t).
    """
    x = finite_1d(samples)
    n = x.size
    if n < 2:
        return np.empty(0, dtype=np.float64)
    x = x - np.mean(x)
    if max_lag is None:
        max_lag = n // 2
    max_lag = max(1, min(int(max_lag), n - 1))
    variance = float(np.mean(x * x))
    if variance <= 0.0 or not np.isfinite(variance):
        return np.full(max_lag + 1, np.nan, dtype=np.float64)
    fft_size = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(x, n=fft_size)
    autocov = np.fft.irfft(f * np.conjugate(f), n=fft_size)[: max_lag + 1]
    autocov /= np.arange(n, n - max_lag - 1, -1, dtype=np.float64)
    rho = autocov / autocov[0]
    return rho


def integrated_autocorrelation_time(
    samples: ArrayLike,
    max_lag: int | None = None,
    window_c: float = 5.0,
) -> AutocorrelationResult:
    """
    Estimate integrated autocorrelation time.
    """
    x = finite_1d(samples)
    n = x.size
    if n == 0:
        return AutocorrelationResult(
            tau_int=np.nan,
            window=0,
            n=0,
            n_eff=np.nan,
            rho1=np.nan,
            mean=np.nan,
            naive_error=np.nan,
            autocorr_error=np.nan,
        )
    mean = float(np.mean(x))
    if n == 1:
        return AutocorrelationResult(
            tau_int=np.nan,
            window=0,
            n=1,
            n_eff=np.nan,
            rho1=np.nan,
            mean=mean,
            naive_error=0.0,
            autocorr_error=np.nan,
        )
    sample_var = float(np.var(x, ddof=1))
    naive_error = float(np.sqrt(sample_var / n))
    rho = autocorrelation_function(x, max_lag=max_lag)
    if rho.size < 2 or not np.isfinite(rho[0]):
        return AutocorrelationResult(
            tau_int=np.nan,
            window=0,
            n=int(n),
            n_eff=np.nan,
            rho1=np.nan,
            mean=mean,
            naive_error=naive_error,
            autocorr_error=np.nan,
        )
    window_c = max(float(window_c), 1.0)
    tau_int = 0.5
    window = 0
    for lag in range(1, rho.size):
        rho_lag = float(rho[lag])
        if not np.isfinite(rho_lag):
            break
        if rho_lag <= 0.0:
            break
        tau_int += rho_lag
        window = lag
        if lag >= window_c * tau_int:
            break
    tau_int = max(float(tau_int), 0.5)
    n_eff = float(n / (2.0 * tau_int))
    autocorr_error = float(np.sqrt(sample_var * 2.0 * tau_int / n))
    rho1 = float(rho[1]) if rho.size > 1 else np.nan
    return AutocorrelationResult(
        tau_int=tau_int,
        window=int(window),
        n=int(n),
        n_eff=n_eff,
        rho1=rho1,
        mean=mean,
        naive_error=naive_error,
        autocorr_error=autocorr_error,
    )