#!/usr/bin/env python3
"""
Load GPU simulation outputs produced by simulation.py, compute observables
with jackknife errors, and plot them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

RUN_FILE_GLOB = "*_L*.npz"


# ============================================================
# Statistics / observables
# ============================================================


def jackknife_blocks(samples, n_bins=20):
    """Return block jackknife mean and error for a 1D series."""
    samples = np.asarray(samples, dtype=np.float64)
    n = len(samples)

    nb = min(int(n_bins), n // 2)
    if nb < 2:
        mean = float(np.mean(samples)) if n > 0 else np.nan
        err = float(np.std(samples, ddof=1) / np.sqrt(max(n, 1))) if n > 1 else 0.0
        return mean, err

    bsize = n // nb
    n_use = nb * bsize
    blocks = samples[:n_use].reshape(nb, bsize)
    block_means = np.mean(blocks, axis=1)

    total = np.sum(block_means)
    jk = np.zeros(nb, dtype=np.float64)
    for i in range(nb):
        jk[i] = (total - block_means[i]) / (nb - 1)

    mean = float(np.mean(jk))
    var = (nb - 1) / nb * np.sum((jk - mean) ** 2)
    return mean, float(np.sqrt(var))


def jackknife_from_block_means(block_means):
    """Return jackknife mean/error from precomputed contiguous block means."""
    block_means = np.asarray(block_means, dtype=np.float64)
    nb = len(block_means)
    if nb == 0:
        return np.nan, np.nan
    if nb == 1:
        return float(block_means[0]), 0.0

    total = np.sum(block_means)
    jk = np.zeros(nb, dtype=np.float64)
    for i in range(nb):
        jk[i] = (total - block_means[i]) / (nb - 1)

    mean = float(np.mean(jk))
    var = (nb - 1) / nb * np.sum((jk - mean) ** 2)
    return mean, float(np.sqrt(var))


def integrated_autocorrelation_time(
    samples,
    max_lag: int | None = None,
    window_c: float = 5.0,
) -> dict[str, float | int]:
    """Estimate integrated autocorrelation time for one observable series."""
    samples = np.asarray(samples, dtype=np.float64)
    samples = samples[np.isfinite(samples)]
    n = len(samples)
    if n < 3:
        return {
            "tau": np.nan,
            "window": 0,
            "n": n,
            "n_eff": np.nan,
            "rho1": np.nan,
        }

    x = samples - np.mean(samples)
    variance = float(np.mean(x * x))
    if variance <= 0.0 or not np.isfinite(variance):
        return {
            "tau": np.nan,
            "window": 0,
            "n": n,
            "n_eff": np.nan,
            "rho1": np.nan,
        }

    if max_lag is None:
        max_lag = n // 2
    max_lag = max(1, min(int(max_lag), n - 1))
    window_c = max(float(window_c), 1.0)

    fft_size = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(x, n=fft_size)
    autocov = np.fft.irfft(f * np.conjugate(f), n=fft_size)[: max_lag + 1]
    autocov /= np.arange(n, n - max_lag - 1, -1, dtype=np.float64)
    rho = autocov / autocov[0]

    tau = 0.5
    window = 0
    for lag in range(1, max_lag + 1):
        rho_lag = float(rho[lag])
        if not np.isfinite(rho_lag) or rho_lag <= 0.0:
            break
        tau += rho_lag
        window = lag
        if lag >= window_c * tau:
            break

    tau = max(float(tau), 0.5)
    return {
        "tau": tau,
        "window": int(window),
        "n": int(n),
        "n_eff": float(n / (2.0 * tau)),
        "rho1": float(rho[1]) if max_lag >= 1 else np.nan,
    }


def _json_scalar_load(value):
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, (bytes, np.bytes_)):
        value = value.decode("utf-8")
    return json.loads(str(value))


def _load_optional_array(data, key, fallback=None):
    if key in data.files:
        return data[key]
    return fallback


def _as_observable_matrix(values, R: int, dtype=np.float64) -> np.ndarray:
    """Normalize saved observable arrays to shape (R, n_measurements)."""
    arr = np.asarray(values, dtype=dtype)
    if arr.ndim == 2 and arr.shape[0] == R:
        return arr
    if arr.size == 0:
        return np.empty((R, 0), dtype=dtype)
    if arr.ndim == 1 and R == 1:
        return arr.reshape(1, -1)
    raise ValueError(
        f"Expected observable array with shape ({R}, n), got {arr.shape}."
    )


def _finite_curve(temps, values, errors):
    temps = np.asarray(temps, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    errors = np.asarray(errors, dtype=np.float64)
    mask = np.isfinite(temps) & np.isfinite(values) & np.isfinite(errors)
    return temps[mask], values[mask], errors[mask]


def _sorted_unique_curve(temps, values, errors):
    temps = np.asarray(temps, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    errors = np.asarray(errors, dtype=np.float64)
    order = np.argsort(temps)
    temps = temps[order]
    values = values[order]
    errors = errors[order]
    temps, unique_idx = np.unique(temps, return_index=True)
    return temps, values[unique_idx], errors[unique_idx]


def _common_temperature_grid(temps_a, temps_b):
    lo = max(float(np.min(temps_a)), float(np.min(temps_b)))
    hi = min(float(np.max(temps_a)), float(np.max(temps_b)))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        return np.empty(0, dtype=np.float64)

    grid = np.unique(np.concatenate((temps_a, temps_b, np.array([lo, hi]))))
    return grid[(grid >= lo) & (grid <= hi)]


def _piecewise_linear_roots(temps, diff):
    temps = np.asarray(temps, dtype=np.float64)
    diff = np.asarray(diff, dtype=np.float64)
    finite = np.isfinite(temps) & np.isfinite(diff)
    temps = temps[finite]
    diff = diff[finite]
    if temps.size < 2:
        return []

    candidates = []
    for i in range(temps.size - 1):
        t0 = float(temps[i])
        t1 = float(temps[i + 1])
        d0 = float(diff[i])
        d1 = float(diff[i + 1])
        if d0 == 0.0:
            candidates.append((t0, abs(d0), i))
        if d0 * d1 > 0.0:
            continue
        if d1 == d0:
            root = 0.5 * (t0 + t1)
        else:
            root = t0 - d0 * (t1 - t0) / (d1 - d0)
        if t0 <= root <= t1:
            candidates.append((float(root), min(abs(d0), abs(d1)), i))

    if diff[-1] == 0.0:
        candidates.append((float(temps[-1]), 0.0, temps.size - 1))
    return candidates


def _piecewise_linear_root(temps, diff, target=None):
    candidates = _piecewise_linear_roots(temps, diff)
    if not candidates:
        return None
    if target is not None and np.isfinite(target):
        return min(candidates, key=lambda item: abs(item[0] - target))
    return min(candidates, key=lambda item: item[1])


def _curve_curve_crossing_temperature(temps_a, values_a, temps_b, values_b, target=None):
    grid = _common_temperature_grid(temps_a, temps_b)
    if grid.size < 2:
        return None
    diff = np.interp(grid, temps_a, values_a) - np.interp(grid, temps_b, values_b)
    root = _piecewise_linear_root(grid, diff, target=target)
    return None if root is None else root[0]


def _curve_line_crossing_temperature(temps, values, line_values, target=None):
    root = _piecewise_linear_root(temps, values - line_values, target=target)
    return None if root is None else root[0]


def _bootstrap_error(central, roots):
    roots = np.asarray(roots, dtype=np.float64)
    roots = roots[np.isfinite(roots)]
    if roots.size < 2:
        return np.nan, int(roots.size)

    # Keep the same crossing branch when noisy resamples create extra roots.
    spread = np.abs(roots - central)
    keep = spread <= np.nanpercentile(spread, 95.0)
    kept = roots[keep]
    if kept.size < 2:
        kept = roots
    return float(np.std(kept, ddof=1)), int(kept.size)


def estimate_largest_L_binder_crossing(
    temps_by_L,
    obs_by_L,
    *,
    crossing_target=1.36,
    n_bootstrap=5000,
    rng_seed=12345,
):
    sizes = sorted(obs_by_L.keys())
    if len(sizes) < 2:
        return {
            "available": False,
            "reason": "need at least two system sizes",
        }

    L_a, L_b = sizes[-2], sizes[-1]
    temps_a, values_a, errors_a = _finite_curve(
        temps_by_L[L_a], obs_by_L[L_a]["U4"], obs_by_L[L_a]["U4_err"]
    )
    temps_b, values_b, errors_b = _finite_curve(
        temps_by_L[L_b], obs_by_L[L_b]["U4"], obs_by_L[L_b]["U4_err"]
    )
    if temps_a.size < 2 or temps_b.size < 2:
        return {
            "available": False,
            "L_pair": (L_a, L_b),
            "reason": "need at least two finite Binder points for each size",
        }

    temps_a, values_a, errors_a = _sorted_unique_curve(temps_a, values_a, errors_a)
    temps_b, values_b, errors_b = _sorted_unique_curve(temps_b, values_b, errors_b)
    T_cross = _curve_curve_crossing_temperature(
        temps_a,
        values_a,
        temps_b,
        values_b,
        target=crossing_target,
    )
    if T_cross is None:
        return {
            "available": False,
            "L_pair": (L_a, L_b),
            "reason": "no sign-changing Binder crossing in the shared T range",
        }

    rng = np.random.default_rng(rng_seed)
    errors_a = np.maximum(errors_a, 0.0)
    errors_b = np.maximum(errors_b, 0.0)
    roots = []
    for _ in range(int(n_bootstrap)):
        sample_a = values_a + rng.normal(0.0, errors_a)
        sample_b = values_b + rng.normal(0.0, errors_b)
        root = _curve_curve_crossing_temperature(
            temps_a,
            sample_a,
            temps_b,
            sample_b,
            target=T_cross,
        )
        if root is not None:
            roots.append(root)

    T_err, n_used = _bootstrap_error(T_cross, roots)
    U4_a = float(np.interp(T_cross, temps_a, values_a))
    U4_b = float(np.interp(T_cross, temps_b, values_b))
    return {
        "available": True,
        "L_pair": (L_a, L_b),
        "T": float(T_cross),
        "T_err": T_err,
        "value": 0.5 * (U4_a + U4_b),
        "target": float(crossing_target),
        "n_bootstrap": n_used,
    }


def estimate_largest_L_bkt_intersection(
    temps_by_L,
    obs_by_L,
    *,
    n_bootstrap=5000,
    rng_seed=67890,
):
    if not obs_by_L:
        return {
            "available": False,
            "reason": "no system sizes available",
        }

    L = max(obs_by_L.keys())
    temps, values, errors = _finite_curve(
        temps_by_L[L], obs_by_L[L]["Y"], obs_by_L[L]["Y_err"]
    )
    if temps.size < 2:
        return {
            "available": False,
            "L": L,
            "reason": "need at least two finite helicity points",
        }

    temps, values, errors = _sorted_unique_curve(temps, values, errors)
    bkt_values = 2.0 * temps / np.pi
    T_cross = _curve_line_crossing_temperature(temps, values, bkt_values)
    if T_cross is None:
        return {
            "available": False,
            "L": L,
            "reason": "no sign-changing intersection with 2T/pi",
        }

    rng = np.random.default_rng(rng_seed)
    errors = np.maximum(errors, 0.0)
    roots = []
    for _ in range(int(n_bootstrap)):
        sample = values + rng.normal(0.0, errors)
        root = _curve_line_crossing_temperature(temps, sample, bkt_values)
        if root is not None:
            roots.append(root)

    T_err, n_used = _bootstrap_error(T_cross, roots)
    return {
        "available": True,
        "L": L,
        "T": float(T_cross),
        "T_err": T_err,
        "value": float(2.0 * T_cross / np.pi),
        "n_bootstrap": n_used,
    }


def _derive_pt_transport_stats(label_positions, record_stride=1):
    label_positions = np.asarray(label_positions)
    record_stride = max(1, int(record_stride))
    if label_positions.ndim != 2 or label_positions.size == 0:
        return {
            "hit_low": np.zeros(0, dtype=np.bool_),
            "hit_high": np.zeros(0, dtype=np.bool_),
            "hit_both_edges": np.zeros(0, dtype=np.bool_),
            "commute_counts": np.zeros(0, dtype=np.int64),
            "round_trip_counts": np.zeros(0, dtype=np.int64),
            "round_trip_durations": np.zeros(0, dtype=np.int64),
        }

    R = label_positions.shape[1]
    hit_low = np.any(label_positions == 0, axis=0)
    hit_high = np.any(label_positions == (R - 1), axis=0)
    commute_counts = np.zeros(R, dtype=np.int64)
    round_trip_counts = np.zeros(R, dtype=np.int64)
    state = np.zeros(R, dtype=np.int8)
    last_t = -np.ones(R, dtype=np.int64)
    round_trip_durations = []

    for rec_idx, row in enumerate(label_positions):
        t_index = rec_idx * record_stride
        for lab in range(R):
            p = int(row[lab])
            st = int(state[lab])

            if st == 0:
                if p == 0:
                    state[lab] = 1
                    last_t[lab] = t_index
                elif p == R - 1:
                    state[lab] = 2
                    last_t[lab] = t_index
                continue

            if st == 1:
                if p == R - 1:
                    commute_counts[lab] += 1
                    state[lab] = 3
                continue

            if st == 2:
                if p == 0:
                    commute_counts[lab] += 1
                    state[lab] = 4
                continue

            if st == 3:
                if p == 0 and last_t[lab] >= 0:
                    round_trip_counts[lab] += 1
                    round_trip_durations.append(int(t_index - last_t[lab]))
                    last_t[lab] = t_index
                    state[lab] = 1
                continue

            if p == R - 1 and last_t[lab] >= 0:
                round_trip_counts[lab] += 1
                round_trip_durations.append(int(t_index - last_t[lab]))
                last_t[lab] = t_index
                state[lab] = 2

    return {
        "hit_low": hit_low,
        "hit_high": hit_high,
        "hit_both_edges": hit_low & hit_high,
        "commute_counts": commute_counts,
        "round_trip_counts": round_trip_counts,
        "round_trip_durations": np.asarray(round_trip_durations, dtype=np.int64),
    }


def compute_observables(
    energies,
    mags,
    helicities,
    temps,
    L,
    n_bins=20,
    energy_block_means=None,
    energy2_block_means=None,
    mag_abs_block_means=None,
    mag2_block_means=None,
    mag4_block_means=None,
):
    """
    Compute:
      e, |m_z2|, C, chi_z2, U4_z2, Y
    """
    temps = np.asarray(temps, dtype=np.float64)

    R = len(temps)
    energies = _as_observable_matrix(energies, R)
    mags = _as_observable_matrix(mags, R)
    helicities = _as_observable_matrix(helicities, R)
    n_meas = min(energies.shape[1], mags.shape[1])
    N = int(L) * int(L)
    betas = 1.0 / temps

    energy_block_means = (
        None
        if energy_block_means is None
        else np.asarray(energy_block_means, dtype=np.float64)
    )
    energy2_block_means = (
        None
        if energy2_block_means is None
        else np.asarray(energy2_block_means, dtype=np.float64)
    )
    mag_abs_block_means = (
        None
        if mag_abs_block_means is None
        else np.asarray(mag_abs_block_means, dtype=np.float64)
    )
    mag2_block_means = (
        None
        if mag2_block_means is None
        else np.asarray(mag2_block_means, dtype=np.float64)
    )
    mag4_block_means = (
        None
        if mag4_block_means is None
        else np.asarray(mag4_block_means, dtype=np.float64)
    )
    have_compact_blocks = all(
        arr is not None and arr.ndim == 2 and arr.shape[0] == R and arr.shape[1] > 0
        for arr in (
            energy_block_means,
            energy2_block_means,
            mag_abs_block_means,
            mag2_block_means,
            mag4_block_means,
        )
    )
    use_compact_blocks = n_meas <= 0 and have_compact_blocks

    e = np.full(R, np.nan)
    e_err = np.full(R, np.nan)
    m_abs = np.full(R, np.nan)
    m_abs_err = np.full(R, np.nan)
    C = np.full(R, np.nan)
    C_err = np.full(R, np.nan)
    chi = np.full(R, np.nan)
    chi_err = np.full(R, np.nan)
    U4 = np.full(R, np.nan)
    U4_err = np.full(R, np.nan)
    Y = np.full(R, np.nan)
    Y_err = np.full(R, np.nan)

    for r in range(R):
        beta = betas[r]

        Y_series = helicities[r]
        Y_series = Y_series[np.isfinite(Y_series)]
        Y[r], Y_err[r] = jackknife_blocks(Y_series, n_bins=n_bins)

        if use_compact_blocks:
            E_block = energy_block_means[r]
            E2_block = energy2_block_means[r]
            Mabs_block = mag_abs_block_means[r]
            M2_block = mag2_block_means[r]
            M4_block = mag4_block_means[r]
            nb = len(E_block)

            e[r], e_err[r] = jackknife_from_block_means(E_block / N)
            m_abs[r], m_abs_err[r] = jackknife_from_block_means(Mabs_block / N)

            if nb == 1:
                E_mean = float(E_block[0])
                E2_mean = float(E2_block[0])
                Mabs_mean = float(Mabs_block[0])
                M2_mean = float(M2_block[0])
                M4_mean = float(M4_block[0])
                C[r] = (beta * beta / N) * (E2_mean - E_mean * E_mean)
                chi[r] = (beta / N) * (M2_mean - Mabs_mean * Mabs_mean)
                U4[r] = (
                    1.0 - M4_mean / (3.0 * M2_mean * M2_mean)
                    if M2_mean > 0
                    else 0.0
                )
                C_err[r] = 0.0
                chi_err[r] = 0.0
                U4_err[r] = 0.0
                continue

            C_jk = np.zeros(nb, dtype=np.float64)
            chi_jk = np.zeros(nb, dtype=np.float64)
            U4_jk = np.zeros(nb, dtype=np.float64)
            sum_E = np.sum(E_block)
            sum_E2 = np.sum(E2_block)
            sum_Mabs = np.sum(Mabs_block)
            sum_M2 = np.sum(M2_block)
            sum_M4 = np.sum(M4_block)

            for i in range(nb):
                E_mean = (sum_E - E_block[i]) / (nb - 1)
                E2_mean = (sum_E2 - E2_block[i]) / (nb - 1)
                Mabs_mean = (sum_Mabs - Mabs_block[i]) / (nb - 1)
                M2_mean = (sum_M2 - M2_block[i]) / (nb - 1)
                M4_mean = (sum_M4 - M4_block[i]) / (nb - 1)

                C_jk[i] = (beta * beta / N) * (E2_mean - E_mean * E_mean)
                chi_jk[i] = (beta / N) * (M2_mean - Mabs_mean * Mabs_mean)
                U4_jk[i] = (
                    1.0 - M4_mean / (3.0 * M2_mean * M2_mean) if M2_mean > 0 else 0.0
                )

            C[r] = float(np.mean(C_jk))
            chi[r] = float(np.mean(chi_jk))
            U4[r] = float(np.mean(U4_jk))
            C_err[r] = float(np.sqrt((nb - 1) / nb * np.sum((C_jk - C[r]) ** 2)))
            chi_err[r] = float(np.sqrt((nb - 1) / nb * np.sum((chi_jk - chi[r]) ** 2)))
            U4_err[r] = float(np.sqrt((nb - 1) / nb * np.sum((U4_jk - U4[r]) ** 2)))
            continue

        if n_meas <= 0:
            continue

        E_series = energies[r, :n_meas]
        M_series = mags[r, :n_meas]
        e_vals = E_series / N
        mabs_vals = np.abs(M_series) / N

        e[r], e_err[r] = jackknife_blocks(e_vals, n_bins=n_bins)
        m_abs[r], m_abs_err[r] = jackknife_blocks(mabs_vals, n_bins=n_bins)

        nb = min(int(n_bins), n_meas // 2)
        if nb < 2:
            E_mean = float(np.mean(E_series))
            E2_mean = float(np.mean(E_series ** 2))
            M2_mean = float(np.mean(M_series ** 2))
            M4_mean = float(np.mean(M_series ** 4))
            Mabs_mean = float(np.mean(np.abs(M_series)))

            C[r] = (beta * beta / N) * (E2_mean - E_mean * E_mean)
            chi[r] = (beta / N) * (M2_mean - Mabs_mean * Mabs_mean)
            U4[r] = (
                1.0 - M4_mean / (3.0 * M2_mean * M2_mean)
                if M2_mean > 0
                else 0.0
            )

            C_err[r] = 0.0
            chi_err[r] = 0.0
            U4_err[r] = 0.0
            continue

        bsize = n_meas // nb
        n_use = nb * bsize
        E_blocks = E_series[:n_use].reshape(nb, bsize)
        M_blocks = M_series[:n_use].reshape(nb, bsize)

        C_jk = np.zeros(nb, dtype=np.float64)
        chi_jk = np.zeros(nb, dtype=np.float64)
        U4_jk = np.zeros(nb, dtype=np.float64)

        E_block_means = np.mean(E_blocks, axis=1)
        E2_block_means = np.mean(E_blocks ** 2, axis=1)
        M2_block_means = np.mean(M_blocks ** 2, axis=1)
        M4_block_means = np.mean(M_blocks ** 4, axis=1)
        Mabs_block_means = np.mean(np.abs(M_blocks), axis=1)

        sum_E = np.sum(E_block_means)
        sum_E2 = np.sum(E2_block_means)
        sum_M2 = np.sum(M2_block_means)
        sum_M4 = np.sum(M4_block_means)
        sum_Mabs = np.sum(Mabs_block_means)

        for i in range(nb):
            E_mean = (sum_E - E_block_means[i]) / (nb - 1)
            E2_mean = (sum_E2 - E2_block_means[i]) / (nb - 1)
            M2_mean = (sum_M2 - M2_block_means[i]) / (nb - 1)
            M4_mean = (sum_M4 - M4_block_means[i]) / (nb - 1)
            Mabs_mean = (sum_Mabs - Mabs_block_means[i]) / (nb - 1)

            C_jk[i] = (beta * beta / N) * (E2_mean - E_mean * E_mean)
            chi_jk[i] = (beta / N) * (M2_mean - Mabs_mean * Mabs_mean)
            U4_jk[i] = (
                1.0 - M4_mean / (3.0 * M2_mean * M2_mean)
                if M2_mean > 0
                else 0.0
            )

        C[r] = float(np.mean(C_jk))
        chi[r] = float(np.mean(chi_jk))
        U4[r] = float(np.mean(U4_jk))

        C_err[r] = float(np.sqrt((nb - 1) / nb * np.sum((C_jk - C[r]) ** 2)))
        chi_err[r] = float(np.sqrt((nb - 1) / nb * np.sum((chi_jk - chi[r]) ** 2)))
        U4_err[r] = float(np.sqrt((nb - 1) / nb * np.sum((U4_jk - U4[r]) ** 2)))

    return {
        "e": e,
        "e_err": e_err,
        "m_abs": m_abs,
        "m_abs_err": m_abs_err,
        "C": C,
        "C_err": C_err,
        "chi": chi,
        "chi_err": chi_err,
        "U4": U4,
        "U4_err": U4_err,
        "Y": Y,
        "Y_err": Y_err,
    }


# ============================================================
# Plotting
# ============================================================


def plot_multi_L_observable(temps_by_L, obs_by_L, field, ylabel, title):
    plt.figure()
    plotted = False
    for L in sorted(obs_by_L.keys()):
        obs = obs_by_L[L]
        temps, values, errors = _finite_curve(
            temps_by_L[L],
            obs[field],
            obs[field + "_err"],
        )
        if temps.size == 0:
            continue
        plt.errorbar(
            temps,
            values,
            yerr=errors,
            fmt="o-",
            markersize=3,
            capsize=3,
            label=fr"$L={L}$",
        )
        plotted = True
    if not plotted:
        plt.close()
        print(f"No finite data available for {title}; skipping.")
        return
    plt.xlabel(r"$T$")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()



def plot_multi_L_binder(temps_by_L, obs_by_L, crossing=None):
    plt.figure()
    plotted = False
    for L in sorted(obs_by_L.keys()):
        obs = obs_by_L[L]
        temps, values, errors = _finite_curve(
            temps_by_L[L],
            obs["U4"],
            obs["U4_err"],
        )
        if temps.size == 0:
            continue
        plt.errorbar(
            temps,
            values,
            yerr=errors,
            fmt="o-",
            markersize=3,
            capsize=3,
            label=fr"$L={L}$",
        )
        plotted = True
    if not plotted:
        plt.close()
        print("No finite Binder data available; skipping.")
        return
    plt.xlabel(r"$T$")
    plt.ylabel(r"$U_4^{(Z_2)}$")
    plt.title(r"$Z_2$ Binder cumulant for different system sizes")
    plt.legend()
    plt.tight_layout()
    plt.show()



def plot_multi_L_helicity(temps_by_L, obs_by_L, intersection=None):
    plt.figure()
    all_temps = []
    plotted = False
    for L in sorted(obs_by_L.keys()):
        obs = obs_by_L[L]
        temps, values, errors = _finite_curve(
            temps_by_L[L],
            obs["Y"],
            obs["Y_err"],
        )
        if temps.size == 0:
            continue
        plt.errorbar(
            temps,
            values,
            yerr=errors,
            fmt="o-",
            capsize=3,
            markersize=3,
            label=fr"$L={L}$",
        )
        all_temps.append(temps)
        plotted = True
    if not plotted:
        plt.close()
        print("No finite helicity data available; skipping.")
        return
    t_union = np.unique(np.concatenate(all_temps))
    plt.plot(t_union, 2.0 * t_union / np.pi, "--", lw=1.2, label=r"$2T/\pi$")
    plt.xlabel(r"$T$")
    plt.ylabel(r"$\Upsilon$")
    plt.title("Helicity modulus and BKT line")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_multi_L_swap_acceptance(temps_by_L, swap_stats_by_L):
    plt.figure()
    plotted = False
    for L in sorted(swap_stats_by_L.keys()):
        temps = np.asarray(temps_by_L[L], dtype=np.float64)
        if len(temps) < 2:
            continue
        mid_T = 0.5 * (temps[:-1] + temps[1:])
        stats = swap_stats_by_L[L]
        acc = stats["swap_acceptance"]
        att = stats["swap_attempts"]
        acc_prob = acc / np.maximum(att, 1)
        plt.plot(mid_T, acc_prob, "o-", label=fr"$L={L}$")
        plotted = True
    if not plotted:
        plt.close()
        print("No swap-acceptance data available; skipping.")
        return
    plt.xlabel(r"midpoint $T$")
    plt.ylabel("swap acceptance")
    plt.title("Parallel tempering swap acceptance")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.show()



def plot_replica_random_walk(
    label_positions,
    labels_to_plot=None,
    title="Replica random walk",
):
    label_positions = np.asarray(label_positions)
    if label_positions.size == 0:
        print("No label positions recorded (try lowering record_stride).")
        return

    n_records, R = label_positions.shape
    t = np.arange(n_records)

    if labels_to_plot is None:
        k = min(6, R)
        labels_to_plot = list(range(k))

    plt.figure()
    for lab in labels_to_plot:
        plt.plot(t, label_positions[:, lab], "-", label=fr"label {lab}")
    plt.xlabel("record index")
    plt.ylabel("temperature slot")
    plt.title(title)
    plt.ylim(-0.5, R - 0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()



def plot_round_trip_durations(durations, title="Round-trip durations"):
    durations = np.asarray(durations)
    if durations.size == 0:
        print("No round trips detected (try longer runs or better ladder).")
        return
    plt.figure()
    bins = min(40, max(10, int(np.sqrt(len(durations)))))
    plt.hist(durations, bins=bins)
    plt.xlabel("duration (swap-attempt time steps)")
    plt.ylabel("count")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ============================================================
# Loading and orchestration
# ============================================================


def load_run(path: Path) -> tuple[int, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        L = int(data["L"])
        temps = data["temps"]
        R = len(temps)
        params = (
            _json_scalar_load(data["params_json"])
            if "params_json" in data.files
            else {}
        )
        energies = _load_optional_array(
            data, "energies", fallback=np.empty((R, 0), dtype=np.float32)
        )
        mags = _load_optional_array(
            data, "mags", fallback=np.empty((R, 0), dtype=np.float32)
        )
        helicities = _load_optional_array(
            data, "helicities", fallback=np.empty((R, 0), dtype=np.float32)
        )
        n_helicity = helicities.shape[1] if np.asarray(helicities).ndim == 2 else 0
        if "derived_observable_measure_sweeps" in data.files:
            derived_sweeps = data["derived_observable_measure_sweeps"]
        elif "helicity_measure_sweeps" in data.files:
            derived_sweeps = data["helicity_measure_sweeps"]
        else:
            stride = int(params.get("derived_observable_stride", 1))
            derived_sweeps = stride * np.arange(n_helicity, dtype=np.int32)
        run = {
            "L": L,
            "temps": temps,
            "betas": _load_optional_array(data, "betas", fallback=1.0 / temps),
            "energies": energies,
            "mags": mags,
            "energy_block_means": _load_optional_array(data, "energy_block_means"),
            "energy2_block_means": _load_optional_array(data, "energy2_block_means"),
            "mag_abs_block_means": _load_optional_array(data, "mag_abs_block_means"),
            "mag2_block_means": _load_optional_array(data, "mag2_block_means"),
            "mag4_block_means": _load_optional_array(data, "mag4_block_means"),
            "observable_block_size": _load_optional_array(data, "observable_block_size"),
            "helicities": helicities,
            "derived_observable_measure_sweeps": derived_sweeps,
            "helicity_measure_sweeps": derived_sweeps,
            "swap_acceptance": _load_optional_array(
                data,
                "swap_acceptance",
                fallback=np.zeros(max(R - 1, 0), dtype=np.int64),
            ),
            "swap_attempts": _load_optional_array(
                data,
                "swap_attempts",
                fallback=np.zeros(max(R - 1, 0), dtype=np.int64),
            ),
            "label_positions": _load_optional_array(
                data, "label_positions", fallback=np.zeros((0, R), dtype=np.int32)
            ),
            "round_trip_counts": _load_optional_array(data, "round_trip_counts"),
            "round_trip_durations": _load_optional_array(data, "round_trip_durations"),
            "commute_counts": _load_optional_array(data, "commute_counts"),
            "hit_low": _load_optional_array(data, "hit_low"),
            "hit_high": _load_optional_array(data, "hit_high"),
            "hit_both_edges": _load_optional_array(data, "hit_both_edges"),
            "params": params,
        }
    return L, run



def _collect_manifest_files(input_dir: Path) -> list[Path]:
    manifest_path = input_dir / "manifest.json"
    if not manifest_path.exists():
        return []

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = []
    for name in manifest.get("files", []):
        path = Path(name)
        if not path.is_absolute():
            path = input_dir / path
        if path.exists():
            files.append(path)
    return sorted(files)


def collect_npz_files(input_dir: Path, file_pattern: str | None = None) -> list[Path]:
    if file_pattern is None:
        files = _collect_manifest_files(input_dir)
        if files:
            return files
        file_pattern = RUN_FILE_GLOB

    files = sorted(path for path in input_dir.glob(file_pattern) if path.is_file())
    if not files:
        raise FileNotFoundError(
            f"No files matching {file_pattern!r} found in {input_dir}"
        )
    return files


def _diagnostic_array(value, fallback, dtype):
    if value is None:
        return fallback
    arr = np.asarray(value, dtype=dtype)
    return arr if arr.size > 0 else fallback


def _measurement_spacing_sweeps(sweeps, fallback=1.0) -> float:
    sweeps = np.asarray(sweeps, dtype=np.float64)
    if sweeps.ndim != 1 or sweeps.size < 2:
        return float(fallback)
    diffs = np.diff(sweeps)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    if diffs.size == 0:
        return float(fallback)
    return float(np.median(diffs))


def _autocorrelation_summary(
    matrix,
    temps,
    *,
    sample_spacing_sweeps: float = 1.0,
    max_lag: int | None = None,
    window_c: float = 5.0,
):
    temps = np.asarray(temps, dtype=np.float64)
    R = len(temps)
    matrix = _as_observable_matrix(matrix, R)
    if matrix.shape[1] < 3:
        return {
            "available": False,
            "reason": "history not saved or too short",
        }

    rows = [
        integrated_autocorrelation_time(
            matrix[r],
            max_lag=max_lag,
            window_c=window_c,
        )
        for r in range(R)
    ]
    tau = np.asarray([row["tau"] for row in rows], dtype=np.float64)
    finite = np.isfinite(tau)
    if not np.any(finite):
        return {
            "available": False,
            "reason": "zero-variance or non-finite history",
        }

    slot = int(np.nanargmax(tau))
    best = rows[slot]
    return {
        "available": True,
        "slot": slot,
        "temperature": float(temps[slot]),
        "tau_samples": float(best["tau"]),
        "tau_sweeps": float(best["tau"] * sample_spacing_sweeps),
        "sample_spacing_sweeps": float(sample_spacing_sweeps),
        "window": int(best["window"]),
        "n": int(best["n"]),
        "n_eff": float(best["n_eff"]),
        "rho1": float(best["rho1"]),
    }


def _worst_autocorrelation_summary(parts):
    available = {
        name: diag for name, diag in parts.items() if diag.get("available", False)
    }
    if not available:
        reasons = sorted({diag.get("reason", "unavailable") for diag in parts.values()})
        return {
            "available": False,
            "reason": "; ".join(reasons) if reasons else "history not saved",
        }

    source, diag = max(
        available.items(),
        key=lambda item: item[1]["tau_samples"],
    )
    merged = dict(diag)
    merged["source"] = source
    return merged


def autocorrelation_diagnostics_for_run(
    run,
    temps,
    L: int,
    *,
    max_lag: int | None = None,
    window_c: float = 5.0,
):
    temps = np.asarray(temps, dtype=np.float64)
    R = len(temps)
    N = int(L) * int(L)

    mags = _as_observable_matrix(run["mags"], R)
    helicities = _as_observable_matrix(run["helicities"], R)
    derived_spacing = _measurement_spacing_sweeps(
        run.get("derived_observable_measure_sweeps"),
        fallback=run.get("params", {}).get("derived_observable_stride", 1.0),
    )
    m = mags / N
    m2 = m * m
    m4 = m2 * m2

    return {
        "binder": _worst_autocorrelation_summary(
            {
                "m2": _autocorrelation_summary(
                    m2,
                    temps,
                    sample_spacing_sweeps=1.0,
                    max_lag=max_lag,
                    window_c=window_c,
                ),
                "m4": _autocorrelation_summary(
                    m4,
                    temps,
                    sample_spacing_sweeps=1.0,
                    max_lag=max_lag,
                    window_c=window_c,
                ),
            }
        ),
        "helicity": _autocorrelation_summary(
            helicities,
            temps,
            sample_spacing_sweeps=derived_spacing,
            max_lag=max_lag,
            window_c=window_c,
        ),
    }


def print_autocorrelation_diagnostics(diag_by_L):
    print("Autocorrelation diagnostics:")
    for L in sorted(diag_by_L.keys()):
        print(f"  L={L}:")
        for name, diag in diag_by_L[L].items():
            if not diag["available"]:
                print(f"    {name}: unavailable ({diag['reason']})")
                continue
            print(
                f"    {name}: "
                f"tau_int={diag['tau_samples']:.2f} samples "
                f"({diag['tau_sweeps']:.2f} sweeps), "
                f"worst T={diag['temperature']:.6g}, "
                f"N_eff~{diag['n_eff']:.1f}/{diag['n']}, "
                f"window={diag['window']}, "
                f"rho1={diag['rho1']:.3f}"
                f"{', source=' + diag['source'] if 'source' in diag else ''}"
            )


def _estimate_pm_text(value, error):
    if not np.isfinite(value):
        return "unavailable"
    if not np.isfinite(error):
        return f"{value:.6g} +/- unavailable"
    return f"{value:.6g} +/- {error:.2g}"


def print_transition_estimates(binder_crossing, bkt_intersection):
    print("Transition estimates from plotted curves:")

    if binder_crossing.get("available", False):
        L_a, L_b = binder_crossing["L_pair"]
        target = binder_crossing.get("target")
        target_text = (
            f", nearest target T={target:.6g}"
            if target is not None and np.isfinite(target)
            else ""
        )
        print(
            f"  Binder crossing L={L_a} and L={L_b}: "
            f"T={_estimate_pm_text(binder_crossing['T'], binder_crossing['T_err'])} "
            f"(linear interpolation{target_text}, "
            f"{binder_crossing['n_bootstrap']} bootstrap roots)"
        )
    else:
        pair = binder_crossing.get("L_pair")
        pair_text = f" L={pair[0]} and L={pair[1]}" if pair else ""
        print(f"  Binder crossing{pair_text}: unavailable ({binder_crossing['reason']})")

    if bkt_intersection.get("available", False):
        print(
            f"  Helicity/BKT intersection L={bkt_intersection['L']}: "
            f"T={_estimate_pm_text(bkt_intersection['T'], bkt_intersection['T_err'])} "
            f"(linear interpolation, {bkt_intersection['n_bootstrap']} bootstrap roots)"
        )
    else:
        size = bkt_intersection.get("L")
        size_text = f" L={size}" if size is not None else ""
        print(
            f"  Helicity/BKT intersection{size_text}: "
            f"unavailable ({bkt_intersection['reason']})"
        )


def main(
    input_dir: str | Path,
    n_bins: int = 20,
    file_pattern: str | None = None,
    tau_max_lag: int | None = None,
    tau_window_c: float = 5.0,
    binder_crossing_target: float = 1.36,
):
    input_dir = Path(input_dir)
    files = collect_npz_files(input_dir, file_pattern=file_pattern)

    temps_by_L: dict[int, np.ndarray] = {}
    obs_by_L: dict[int, dict] = {}
    swap_stats_by_L: dict[int, dict] = {}
    pt_diag_by_L: dict[int, dict] = {}
    autocorr_diag_by_L: dict[int, dict] = {}
    files_by_L: dict[int, Path] = {}
    for file in files:
        L, run = load_run(file)
        temps = np.asarray(run["temps"], dtype=np.float64)
        obs = compute_observables(
            run["energies"],
            run["mags"],
            run["helicities"],
            temps,
            L=L,
            n_bins=n_bins,
            energy_block_means=run["energy_block_means"],
            energy2_block_means=run["energy2_block_means"],
            mag_abs_block_means=run["mag_abs_block_means"],
            mag2_block_means=run["mag2_block_means"],
            mag4_block_means=run["mag4_block_means"],
        )

        temps_by_L[L] = temps
        obs_by_L[L] = obs
        files_by_L[L] = file
        autocorr_diag_by_L[L] = autocorrelation_diagnostics_for_run(
            run,
            temps,
            L=L,
            max_lag=tau_max_lag,
            window_c=tau_window_c,
        )
        swap_stats_by_L[L] = {
            "swap_acceptance": run["swap_acceptance"],
            "swap_attempts": run["swap_attempts"],
        }
        pt_diag_by_L[L] = {
            "label_positions": run["label_positions"],
            "round_trip_counts": run["round_trip_counts"],
            "round_trip_durations": run["round_trip_durations"],
            "commute_counts": run["commute_counts"],
            "hit_low": run["hit_low"],
            "hit_high": run["hit_high"],
            "hit_both_edges": run["hit_both_edges"],
            "params": run["params"],
        }

    binder_crossing = estimate_largest_L_binder_crossing(
        temps_by_L,
        obs_by_L,
        crossing_target=binder_crossing_target,
    )
    bkt_intersection = estimate_largest_L_bkt_intersection(temps_by_L, obs_by_L)
    print_transition_estimates(binder_crossing, bkt_intersection)

    plot_multi_L_observable(
        temps_by_L,
        obs_by_L,
        "e",
        r"$e$",
        "Energy per site for different system sizes",
    )
    plot_multi_L_observable(
        temps_by_L,
        obs_by_L,
        "m_abs",
        r"$|m_{Z_2}|$",
        r"$Z_2$ magnetization per site",
    )
    plot_multi_L_observable(
        temps_by_L,
        obs_by_L,
        "C",
        r"$C$",
        "Specific heat for different system sizes",
    )
    plot_multi_L_observable(
        temps_by_L,
        obs_by_L,
        "chi",
        r"$\chi_{Z_2}$",
        r"$Z_2$ susceptibility",
    )
    plot_multi_L_helicity(temps_by_L, obs_by_L, intersection=bkt_intersection)
    plot_multi_L_binder(temps_by_L, obs_by_L, crossing=binder_crossing)
    plot_multi_L_swap_acceptance(temps_by_L, swap_stats_by_L)

    print_autocorrelation_diagnostics(autocorr_diag_by_L)

    print("Parallel tempering diagnostics:")
    for L in sorted(pt_diag_by_L.keys()):
        diag = pt_diag_by_L[L]
        pos = np.asarray(diag["label_positions"])
        record_stride = int(diag["params"].get("record_stride", 1))
        derived = _derive_pt_transport_stats(pos, record_stride=record_stride)
        hit_low = (
            np.asarray(diag["hit_low"], dtype=np.bool_)
            if diag["hit_low"] is not None
            else derived["hit_low"]
        )
        hit_high = (
            np.asarray(diag["hit_high"], dtype=np.bool_)
            if diag["hit_high"] is not None
            else derived["hit_high"]
        )
        hit_both = (
            np.asarray(diag["hit_both_edges"], dtype=np.bool_)
            if diag["hit_both_edges"] is not None
            else derived["hit_both_edges"]
        )
        commute_counts = (
            np.asarray(diag["commute_counts"], dtype=np.int64)
            if diag["commute_counts"] is not None
            else derived["commute_counts"]
        )
        round_trip_counts = _diagnostic_array(
            diag["round_trip_counts"],
            derived["round_trip_counts"],
            np.int64,
        )
        n_labels = int(pos.shape[1]) if pos.ndim == 2 else 0
        print(
            f"  L={L}: "
            f"round trips={int(round_trip_counts.sum())}, "
            f"one-way commutes={int(commute_counts.sum())}, "
            f"hit low/high/both={int(hit_low.sum())}/{int(hit_high.sum())}/{int(hit_both.sum())} "
            f"of {n_labels}"
        )

    L_show = int(max(temps_by_L.keys()))
    L_rt_show = max(
        (
            L
            for L, diag in pt_diag_by_L.items()
            if _diagnostic_array(
                diag["round_trip_durations"],
                _derive_pt_transport_stats(
                    diag["label_positions"],
                    record_stride=int(diag["params"].get("record_stride", 1)),
                )["round_trip_durations"],
                np.int64,
            ).size > 0
        ),
        default=L_show,
    )
    if L_rt_show != L_show:
        print(
            f"Showing round-trip plots for L={L_rt_show}; "
            f"L={L_show} has no complete round trips in the recorded window."
        )

    plot_replica_random_walk(
        pt_diag_by_L[L_rt_show]["label_positions"],
        title=fr"Replica random walk (L={L_rt_show})",
    )
    plot_round_trip_durations(
        _diagnostic_array(
            pt_diag_by_L[L_rt_show]["round_trip_durations"],
            _derive_pt_transport_stats(
                pt_diag_by_L[L_rt_show]["label_positions"],
                record_stride=int(
                    pt_diag_by_L[L_rt_show]["params"].get("record_stride", 1)
                ),
            )["round_trip_durations"],
            np.int64,
        ),
        title=fr"Round-trip durations (L={L_rt_show})",
    )

    print("Loaded files:")
    for L in sorted(temps_by_L.keys()):
        print(f"  L={L}: {files_by_L[L]}")



def build_parser():
    p = argparse.ArgumentParser(description="Plot saved GPU parallel-tempering runs")
    p.add_argument("--input-dir", type=str, default="gpu_raw_runs")
    p.add_argument("--n-bins", type=int, default=20)
    p.add_argument(
        "--file-pattern",
        type=str,
        default=None,
        help="Optional glob pattern for run files when no manifest is available.",
    )
    p.add_argument(
        "--tau-max-lag",
        type=int,
        default=None,
        help="Optional maximum lag, in saved samples, for autocorrelation estimates.",
    )
    p.add_argument(
        "--tau-window-c",
        type=float,
        default=5.0,
        help="Self-consistent autocorrelation window factor.",
    )
    p.add_argument(
        "--binder-crossing-target",
        type=float,
        default=1.36,
        help="Temperature used to select the Binder crossing branch.",
    )
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    main(
        args.input_dir,
        n_bins=args.n_bins,
        file_pattern=args.file_pattern,
        tau_max_lag=args.tau_max_lag,
        tau_window_c=args.tau_window_c,
        binder_crossing_target=args.binder_crossing_target,
    )
