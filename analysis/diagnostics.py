from __future__ import annotations
from typing import Any
import numpy as np
from analysis.statistics import integrated_autocorrelation_time

ArrayLike = Any

def safe_rate(
    accepted: ArrayLike,
    attempted: ArrayLike,
) -> np.ndarray:
    """
    Compute accepted / attempted safely.
    Where attempted == 0, the rate is nan.
    """
    accepted = np.asarray(accepted, dtype=np.float64)
    attempted = np.asarray(attempted, dtype=np.float64)
    if accepted.shape != attempted.shape:
        raise ValueError(
            "accepted and attempted must have the same shape. "
            f"Got {accepted.shape} and {attempted.shape}."
        )
    rate = np.full_like(accepted, np.nan, dtype=np.float64)
    mask = attempted > 0.0
    rate[mask] = accepted[mask] / attempted[mask]
    return rate

def compute_swap_diagnostics(
    *,
    swap_acceptance: ArrayLike,
    swap_attempts: ArrayLike,
) -> dict[str, Any]:
    """
    Compute parallel-tempering swap acceptance diagnostics.
    """
    swap_acceptance = np.asarray(swap_acceptance, dtype=np.float64)
    swap_attempts = np.asarray(swap_attempts, dtype=np.float64)
    if swap_acceptance.shape != swap_attempts.shape:
        raise ValueError(
            "swap_acceptance and swap_attempts must have the same shape."
        )
    swap_rate = safe_rate(swap_acceptance, swap_attempts)
    finite = np.isfinite(swap_rate)
    if np.any(finite):
        mean_rate = float(np.mean(swap_rate[finite]))
        min_rate = float(np.min(swap_rate[finite]))
        max_rate = float(np.max(swap_rate[finite]))
    else:
        mean_rate = np.nan
        min_rate = np.nan
        max_rate = np.nan
    return {
        "swap_rate": swap_rate,
        "swap_rate_mean": mean_rate,
        "swap_rate_min": min_rate,
        "swap_rate_max": max_rate,
        "swap_acceptance_total": int(np.sum(swap_acceptance)),
        "swap_attempts_total": int(np.sum(swap_attempts)),
    }

def compute_local_acceptance_diagnostics(
    *,
    local_update_acceptance: ArrayLike,
    local_update_attempts: ArrayLike,
    name: str = "local",
) -> dict[str, Any]:
    """
    Compute local update acceptance diagnostics.
    """
    local_update_acceptance = np.asarray(
        local_update_acceptance,
        dtype=np.float64,
    )
    local_update_attempts = np.asarray(
        local_update_attempts,
        dtype=np.float64,
    )
    rate = safe_rate(
        local_update_acceptance,
        local_update_attempts,
    )
    finite = np.isfinite(rate)
    if np.any(finite):
        mean_rate = float(np.mean(rate[finite]))
        min_rate = float(np.min(rate[finite]))
        max_rate = float(np.max(rate[finite]))
    else:
        mean_rate = np.nan
        min_rate = np.nan
        max_rate = np.nan
    return {
        f"{name}_acceptance_rate": rate,
        f"{name}_acceptance_rate_mean": mean_rate,
        f"{name}_acceptance_rate_min": min_rate,
        f"{name}_acceptance_rate_max": max_rate,
        f"{name}_accepted_total": int(np.sum(local_update_acceptance)),
        f"{name}_attempted_total": int(np.sum(local_update_attempts)),
    }

def compute_energy_drift_diagnostics(
    *,
    energy_drift: ArrayLike | None = None,
    energy_drift_abs_max: ArrayLike | None = None,
    energy_drift_recompute_count: ArrayLike | None = None,
) -> dict[str, Any]:
    """
    Summarize energy drift diagnostics.
    """
    diagnostics: dict[str, Any] = {}
    if energy_drift is not None:
        drift = np.asarray(energy_drift, dtype=np.float64)
        finite = np.isfinite(drift)
        diagnostics["energy_drift"] = drift
        if np.any(finite):
            diagnostics["energy_drift_mean"] = float(np.mean(drift[finite]))
            diagnostics["energy_drift_abs_mean"] = float(
                np.mean(np.abs(drift[finite]))
            )
            diagnostics["energy_drift_abs_max"] = float(
                np.max(np.abs(drift[finite]))
            )
        else:
            diagnostics["energy_drift_mean"] = np.nan
            diagnostics["energy_drift_abs_mean"] = np.nan
            diagnostics["energy_drift_abs_max"] = np.nan
    if energy_drift_abs_max is not None:
        arr = np.asarray(energy_drift_abs_max, dtype=np.float64)
        diagnostics["energy_drift_abs_max_by_walker"] = arr
        finite = np.isfinite(arr)
        diagnostics["energy_drift_abs_max_global"] = (
            float(np.max(arr[finite])) if np.any(finite) else np.nan
        )
    if energy_drift_recompute_count is not None:
        arr = np.asarray(energy_drift_recompute_count)
        diagnostics["energy_drift_recompute_count"] = arr
        diagnostics["energy_drift_recompute_count_total"] = int(np.sum(arr))
    return diagnostics

def derive_pt_transport_stats(
    label_positions: ArrayLike,
    *,
    record_stride: int = 1,
) -> dict[str, np.ndarray]:
    """
    Derive PT transport statistics from walker label positions.
    """
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
    n_records, n_walkers = label_positions.shape
    low = 0
    high = n_walkers - 1
    hit_low = np.any(label_positions == low, axis=0)
    hit_high = np.any(label_positions == high, axis=0)
    commute_counts = np.zeros(n_walkers, dtype=np.int64)
    round_trip_counts = np.zeros(n_walkers, dtype=np.int64)
    round_trip_durations: list[int] = []
    state = np.zeros(n_walkers, dtype=np.int8)
    last_t = -np.ones(n_walkers, dtype=np.int64)
    for rec_idx in range(n_records):
        t_index = rec_idx * record_stride
        row = label_positions[rec_idx]
        for walker in range(n_walkers):
            position = int(row[walker])
            walker_state = int(state[walker])
            if walker_state == 0:
                if position == low:
                    state[walker] = 1
                    last_t[walker] = t_index
                elif position == high:
                    state[walker] = 2
                    last_t[walker] = t_index
                continue
            if walker_state == 1:
                if position == high:
                    commute_counts[walker] += 1
                    state[walker] = 3
                continue
            if walker_state == 2:
                if position == low:
                    commute_counts[walker] += 1
                    state[walker] = 4
                continue
            if walker_state == 3:
                if position == low and last_t[walker] >= 0:
                    round_trip_counts[walker] += 1
                    round_trip_durations.append(
                        int(t_index - last_t[walker])
                    )
                    last_t[walker] = t_index
                    state[walker] = 1
                continue
            if walker_state == 4:
                if position == high and last_t[walker] >= 0:
                    round_trip_counts[walker] += 1
                    round_trip_durations.append(
                        int(t_index - last_t[walker])
                    )
                    last_t[walker] = t_index
                    state[walker] = 2
    return {
        "hit_low": hit_low,
        "hit_high": hit_high,
        "hit_both_edges": hit_low & hit_high,
        "commute_counts": commute_counts,
        "round_trip_counts": round_trip_counts,
        "round_trip_durations": np.asarray(
            round_trip_durations,
            dtype=np.int64,
        ),
    }

def compute_autocorrelation_by_temperature(
    values: ArrayLike,
    *,
    max_lag: int | None = None,
    window_c: float = 5.0,
) -> dict[str, np.ndarray]:
    """
    Compute autocorrelation summaries for each temperature row.
    Input shape:
        (n_temps, n_measurements)
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(
            "values must have shape (n_temps, n_measurements). "
            f"Got {values.shape}."
        )
    n_temps = values.shape[0]
    tau_int = np.full(n_temps, np.nan)
    window = np.zeros(n_temps, dtype=np.int64)
    n = np.zeros(n_temps, dtype=np.int64)
    n_eff = np.full(n_temps, np.nan)
    rho1 = np.full(n_temps, np.nan)
    mean = np.full(n_temps, np.nan)
    naive_error = np.full(n_temps, np.nan)
    autocorr_error = np.full(n_temps, np.nan)
    for r in range(n_temps):
        result = integrated_autocorrelation_time(
            values[r],
            max_lag=max_lag,
            window_c=window_c,
        )
        tau_int[r] = result.tau_int
        window[r] = result.window
        n[r] = result.n
        n_eff[r] = result.n_eff
        rho1[r] = result.rho1
        mean[r] = result.mean
        naive_error[r] = result.naive_error
        autocorr_error[r] = result.autocorr_error
    return {
        "tau_int": tau_int,
        "autocorr_window": window,
        "autocorr_n": n,
        "n_eff": n_eff,
        "rho1": rho1,
        "autocorr_mean": mean,
        "naive_error": naive_error,
        "autocorr_error": autocorr_error,
    }

def compute_run_diagnostics(
    data: dict[str, Any],
    *,
    record_stride: int = 1,
    autocorrelation_keys: list[str] | None = None,
    max_lag: int | None = None,
    window_c: float = 5.0,
) -> dict[str, Any]:
    """
    Compute available diagnostics from one loaded run dictionary.
    """
    diagnostics: dict[str, Any] = {}
    if "swap_acceptance" in data and "swap_attempts" in data:
        diagnostics.update(
            compute_swap_diagnostics(
                swap_acceptance=data["swap_acceptance"],
                swap_attempts=data["swap_attempts"],
            )
        )
    if "local_update_acceptance" in data and "local_update_attempts" in data:
        diagnostics.update(
            compute_local_acceptance_diagnostics(
                local_update_acceptance=data["local_update_acceptance"],
                local_update_attempts=data["local_update_attempts"],
                name="local",
            )
        )
    if "label_positions" in data:
        diagnostics.update(
            derive_pt_transport_stats(
                data["label_positions"],
                record_stride=record_stride,
            )
        )
    energy_drift_keys = {
        "energy_drift",
        "energy_drift_last",
        "energy_drift_abs_max",
        "energy_drift_max",
        "energy_drift_recompute_count",
        "energy_recompute_checks",
    }
    if any(key in data for key in energy_drift_keys):
        diagnostics.update(
            compute_energy_drift_diagnostics(
                energy_drift=data.get(
                    "energy_drift",
                    data.get("energy_drift_last"),
                ),
                energy_drift_abs_max=data.get(
                    "energy_drift_abs_max",
                    data.get("energy_drift_max"),
                ),
                energy_drift_recompute_count=data.get(
                    "energy_drift_recompute_count",
                    data.get("energy_recompute_checks"),
                ),
            )
        )
    if autocorrelation_keys is None:
        autocorrelation_keys = []
    for key in autocorrelation_keys:
        if key not in data:
            continue
        values = np.asarray(data[key])
        if values.ndim != 2:
            continue
        ac = compute_autocorrelation_by_temperature(
            values,
            max_lag=max_lag,
            window_c=window_c,
        )
        for ac_key, ac_value in ac.items():
            diagnostics[f"{key}_{ac_key}"] = ac_value
    return diagnostics
