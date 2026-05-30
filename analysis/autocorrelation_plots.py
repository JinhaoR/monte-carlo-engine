from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from analysis.statistics import autocorrelation_function

FIGSIZE = (7.0, 4.8)
FIG_DPI = 200
MULTI_L_MARKERSIZE = 2.0
MULTI_L_LINEWIDTH = 1.45
MAX_TAGGED_ACF_LAG_SAMPLES = 20000
MAX_TAGGED_ACF_WALKERS_PER_PLOT = 12


def add_autocorrelation_aliases(data: dict[str, Any]) -> dict[str, Any]:
    """
    Add derived history aliases used only by autocorrelation diagnostics.
    """
    out = dict(data)
    if "order_parameter" in out:
        order = np.asarray(out["order_parameter"])
        if order.ndim == 2 and order.shape[1] > 2:
            out.setdefault("z2_order_abs", np.abs(order))
            out.setdefault("z2_order2", order * order)

    for tracked_key in (
        "tracked_order_parameter",
        "tracked_z2_order_parameter",
        "tracked_chirality_order_parameter",
    ):
        if tracked_key not in out:
            continue
        tracked_order = np.asarray(out[tracked_key])
        if tracked_order.ndim == 2 and tracked_order.shape[1] > 2:
            out.setdefault("tracked_z2_order_abs", np.abs(tracked_order))
            out.setdefault("tracked_z2_order2", tracked_order * tracked_order)
            break
    return out


def infer_autocorrelation_keys(data: dict[str, Any]) -> list[str]:
    """
    Choose temperature-slot full histories for autocorrelation analysis.
    """
    candidates = [
        "energies",
        "energy",
        "z2_order_abs",
        "z2_order2",
        "mags",
        "magnetization",
        "nbar",
        "n_bars",
        "helicities",
        "total_amplitude",
    ]
    keys: list[str] = []
    for key in candidates:
        if key not in data:
            continue
        arr = np.asarray(data[key])
        if arr.ndim == 2 and arr.shape[1] > 2:
            keys.append(key)
    return keys


def infer_tagged_autocorrelation_keys(data: dict[str, Any]) -> list[str]:
    """
    Choose tagged-walker histories for autocorrelation analysis.
    """
    candidates = [
        "tracked_energies",
        "tracked_z2_order_abs",
        "tracked_z2_order2",
        "tracked_order_parameter",
        "tracked_z2_order_parameter",
        "tracked_chirality_order_parameter",
        "tracked_helicities",
        "tracked_total_density",
        "tracked_density",
        "tracked_amplitude_imbalance",
    ]
    tracked_walkers = np.asarray(data.get("tracked_walkers", []))
    n_tracked = int(tracked_walkers.size)
    keys: list[str] = []
    for key in candidates:
        if key not in data:
            continue
        arr = np.asarray(data[key])
        if arr.ndim == 2 and arr.shape[0] == n_tracked and arr.shape[1] > 2:
            keys.append(key)
    return keys


def _stride_from_measure_sweeps(values: Any, fallback: int = 1) -> int:
    """
    Infer a positive sweep stride from saved measurement-sweep indices.
    """
    arr = np.asarray(values)
    if arr.ndim != 1 or arr.size < 2:
        return max(1, int(fallback))
    diffs = np.diff(arr.astype(np.int64))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return max(1, int(fallback))
    return max(1, int(round(float(np.median(diffs)))))


def infer_autocorrelation_sample_strides(
    data: dict[str, Any],
    keys: list[str],
    params: dict[str, Any],
) -> dict[str, int]:
    """
    Infer how many sweeps separate saved samples for temperature-slot histories.
    """
    n_measure_sweeps = params.get("n_measure_sweeps")
    try:
        n_measure_sweeps = int(n_measure_sweeps)
    except (TypeError, ValueError):
        n_measure_sweeps = None

    configured_derived_stride = int(params.get("derived_observable_stride", 1))
    derived_sweeps = data.get("derived_observable_measure_sweeps")
    derived_count = None
    derived_stride = configured_derived_stride
    if derived_sweeps is not None:
        derived_arr = np.asarray(derived_sweeps)
        if derived_arr.ndim == 1:
            derived_count = int(derived_arr.size)
            derived_stride = _stride_from_measure_sweeps(
                derived_arr,
                fallback=configured_derived_stride,
            )

    strides: dict[str, int] = {}
    for key in keys:
        arr = np.asarray(data.get(key))
        if arr.ndim != 2:
            continue
        n_samples = int(arr.shape[1])
        if n_measure_sweeps is not None and n_samples == n_measure_sweeps:
            strides[key] = 1
        elif derived_count is not None and n_samples == derived_count:
            strides[key] = derived_stride
        elif n_measure_sweeps and n_samples > 0:
            strides[key] = max(
                1,
                int(round(float(n_measure_sweeps) / float(n_samples))),
            )
        else:
            strides[key] = 1
    return strides


def infer_tagged_autocorrelation_sample_strides(
    data: dict[str, Any],
    keys: list[str],
    params: dict[str, Any],
) -> dict[str, int]:
    """
    Infer sweep stride for tagged-walker histories.
    """
    configured_derived_stride = int(params.get("derived_observable_stride", 1))
    tracked_sweeps = data.get(
        "tracked_observable_measure_sweeps",
        data.get("derived_observable_measure_sweeps"),
    )
    stride = _stride_from_measure_sweeps(
        tracked_sweeps,
        fallback=configured_derived_stride,
    )
    return {key: stride for key in keys}


def attach_autocorrelation_metadata(
    obs: dict[str, Any],
    data: dict[str, Any],
    *,
    autocorrelation_keys: list[str],
    autocorrelation_sample_strides: dict[str, int],
    tagged_autocorrelation_keys: list[str],
    tagged_autocorrelation_sample_strides: dict[str, int],
) -> None:
    """
    Attach autocorrelation metadata and lightweight tagged histories to obs.
    """
    obs["_autocorrelation_keys"] = autocorrelation_keys
    obs["_autocorrelation_sample_strides"] = autocorrelation_sample_strides
    obs["_tagged_autocorrelation_keys"] = tagged_autocorrelation_keys
    obs["_tagged_autocorrelation_sample_strides"] = (
        tagged_autocorrelation_sample_strides
    )
    if "tracked_walkers" in data:
        obs["tracked_walkers"] = data["tracked_walkers"]
    if tagged_autocorrelation_keys:
        obs["_tagged_autocorrelation_histories"] = {
            key: np.asarray(data[key])
            for key in tagged_autocorrelation_keys
            if key in data
        }


def summarize_autocorrelation(obs: dict[str, Any]) -> dict[str, Any]:
    """
    Build JSON-ready autocorrelation summary fragments for one analyzed run.
    """
    tau_summary: dict[str, Any] = {}
    tagged_tau_summary: dict[str, Any] = {}
    for key in sorted(obs):
        if not key.endswith("_tau_int_sweeps"):
            continue
        if key.startswith("tracked_"):
            tagged_tau_summary[key] = _finite_stats(obs[key])
        else:
            tau_summary[key] = _finite_stats(obs[key])

    summary: dict[str, Any] = {}
    if tau_summary:
        summary["autocorrelation"] = tau_summary
        summary["autocorrelation_sample_strides"] = obs.get(
            "_autocorrelation_sample_strides",
            {},
        )
    if tagged_tau_summary:
        summary["tagged_autocorrelation"] = tagged_tau_summary
        summary["tagged_autocorrelation_sample_strides"] = obs.get(
            "_tagged_autocorrelation_sample_strides",
            {},
        )
    return summary


def plot_autocorrelation_diagnostics(
    analyzed_by_L: dict[int, dict[str, Any]],
    plots_dir: Path,
) -> list[Path]:
    """
    Plot all temperature-slot and tagged-walker autocorrelation diagnostics.
    """
    written: list[Path] = []
    written.extend(plot_temperature_autocorrelation_times(analyzed_by_L, plots_dir))
    written.extend(plot_tagged_autocorrelation_times(analyzed_by_L, plots_dir))
    written.extend(plot_tagged_autocorrelation_functions(analyzed_by_L, plots_dir))
    return written


def plot_temperature_autocorrelation_times(
    analyzed_by_L: dict[int, dict[str, Any]],
    plots_dir: Path,
) -> list[Path]:
    """
    Plot temperature-slot tau_int curves produced by diagnostics.
    """
    tau_keys = sorted(
        {
            key
            for obs in analyzed_by_L.values()
            for key in obs
            if key.endswith("_tau_int_sweeps") and not key.startswith("tracked_")
        }
    )
    ylabel = r"Temperature-slot $\tau_{\mathrm{int}}$ (sweeps)"
    if not tau_keys:
        tau_keys = sorted(
            {
                key
                for obs in analyzed_by_L.values()
                for key in obs
                if key.endswith("_tau_int") and not key.startswith("tracked_")
            }
        )
        ylabel = r"Temperature-slot $\tau_{\mathrm{int}}$ (saved samples)"

    written: list[Path] = []
    for key in tau_keys:
        available = [
            int(L)
            for L, obs in analyzed_by_L.items()
            if _has_temperature_curve(obs, key)
        ]
        if not available:
            continue
        colors = _multi_L_colors(available)
        fig, ax = plt.subplots(figsize=FIGSIZE)
        for L in available:
            obs = analyzed_by_L[L]
            _multi_L_plot(
                ax,
                np.asarray(obs["temps"], dtype=np.float64),
                np.asarray(obs[key], dtype=np.float64),
                L=L,
                color=colors.get(int(L)),
            )
        _format_axes(
            ax,
            xlabel=r"Temperature $T$",
            ylabel=ylabel,
            title=autocorrelation_title(key),
        )
        _legend(ax)
        out_path = plots_dir / f"{safe_filename_token(key)}.png"
        finish_figure(fig, out_path)
        written.append(out_path)
    return written


def plot_tagged_autocorrelation_times(
    analyzed_by_L: dict[int, dict[str, Any]],
    plots_dir: Path,
) -> list[Path]:
    """
    Plot tagged-walker tau_int diagnostics for the largest tracked L.
    """
    tau_keys = sorted(
        {
            key
            for obs in analyzed_by_L.values()
            for key in obs
            if key.startswith("tracked_") and key.endswith("_tau_int_sweeps")
        }
    )
    ylabel = r"Tagged-walker $\tau_{\mathrm{int}}$ (sweeps)"
    if not tau_keys:
        tau_keys = sorted(
            {
                key
                for obs in analyzed_by_L.values()
                for key in obs
                if key.startswith("tracked_") and key.endswith("_tau_int")
            }
        )
        ylabel = r"Tagged-walker $\tau_{\mathrm{int}}$ (saved samples)"

    written: list[Path] = []
    for key in tau_keys:
        L = _largest_L_with_tracked_key(analyzed_by_L, key)
        if L is None:
            continue

        obs = analyzed_by_L[L]
        values = np.asarray(obs[key], dtype=np.float64)
        walkers = np.asarray(obs["tracked_walkers"], dtype=np.int64)
        finite = np.isfinite(values)
        if not np.any(finite):
            continue

        fig, ax = plt.subplots(figsize=FIGSIZE)
        _multi_L_plot(
            ax,
            walkers[finite],
            values[finite],
            L=L,
            color=_multi_L_colors([int(L)]).get(int(L)),
        )
        _format_axes(
            ax,
            xlabel="Tracked walker ID",
            ylabel=ylabel,
            title=tagged_autocorrelation_title(key, L=int(L)),
            integer_x=True,
        )
        _legend(ax)
        out_path = plots_dir / f"{safe_filename_token(key)}_L{int(L)}.png"
        finish_figure(fig, out_path)
        written.append(out_path)
    return written


def plot_tagged_autocorrelation_functions(
    analyzed_by_L: dict[int, dict[str, Any]],
    plots_dir: Path,
    *,
    max_lag_samples: int = MAX_TAGGED_ACF_LAG_SAMPLES,
    max_walkers_per_plot: int = MAX_TAGGED_ACF_WALKERS_PER_PLOT,
) -> list[Path]:
    """
    Plot rho(lag) curves for each tracked-walker history.
    """
    written: list[Path] = []
    L = _largest_L_with_tracked_histories(analyzed_by_L)
    if L is None:
        return written
    obs = analyzed_by_L[L]
    histories = obs.get("_tagged_autocorrelation_histories")
    if not isinstance(histories, dict) or not histories:
        return written
    walkers = np.asarray(obs.get("tracked_walkers", []), dtype=np.int64)
    if walkers.size == 0:
        return written
    strides = obs.get("_tagged_autocorrelation_sample_strides", {})
    for key in sorted(histories):
        values = np.asarray(histories[key], dtype=np.float64)
        if values.ndim != 2 or values.shape[0] != walkers.size:
            continue
        if values.shape[1] < 3:
            continue
        stride = max(1, int(strides.get(key, 1)))
        max_lag = min(values.shape[1] - 1, int(max_lag_samples))
        if max_lag < 1:
            continue
        selected_rows = _selected_row_indices(
            walkers.size,
            max_walkers_per_plot=max_walkers_per_plot,
        )
        cmap = plt.get_cmap("viridis")
        fig, ax = plt.subplots(figsize=FIGSIZE)
        for n, row in enumerate(selected_rows):
            rho = autocorrelation_function(values[row], max_lag=max_lag)
            if rho.size == 0 or not np.any(np.isfinite(rho)):
                continue
            lag_sweeps = np.arange(rho.size, dtype=np.float64) * stride
            color = cmap(n / max(1, len(selected_rows) - 1))
            ax.plot(
                lag_sweeps,
                rho,
                linewidth=1.0,
                alpha=0.9,
                color=color,
                label=f"walker {int(walkers[row])}",
            )
        ax.axhline(0.0, color="0.35", linestyle=":", linewidth=0.9)
        _format_axes(
            ax,
            xlabel="Lag (sweeps)",
            ylabel=r"Autocorrelation $\rho(t)$",
            title=tagged_autocorrelation_function_title(key, int(L)),
        )
        _legend(ax)
        out_path = (
            plots_dir
            / f"{safe_filename_token(key)}_autocorrelation_function_L{int(L)}.png"
        )
        finish_figure(fig, out_path)
        written.append(out_path)
    return written


def autocorrelation_title(key: str) -> str:
    """
    Title for temperature-slot autocorrelation diagnostics.
    """
    name = _strip_tau_suffix(key)
    labels = {
        "energies": "Energy",
        "energy": "Energy",
        "z2_order_abs": r"Z2 $|M|$",
        "z2_order2": r"Z2 $M^2$",
        "mags": "Order parameter",
        "magnetization": "Magnetization",
        "nbar": "nbar",
        "n_bars": "nbar",
        "helicities": "Helicity",
        "total_amplitude": "Total amplitude",
    }
    label = labels.get(name, name.replace("_", " ").title())
    return f"Temperature-slot {label} autocorrelation time"


def tagged_autocorrelation_title(key: str, *, L: int | None = None) -> str:
    """
    Title for tagged-walker tau_int diagnostics.
    """
    label = _tagged_label(_strip_tau_suffix(key))
    prefix = f"L={int(L)} " if L is not None else ""
    return f"{prefix}tagged-walker {label} autocorrelation time"


def tagged_autocorrelation_function_title(key: str, L: int) -> str:
    """
    Title for tagged-walker rho(lag) diagnostics.
    """
    return f"L={L} tagged-walker {_tagged_label(key)} autocorrelation function"


def _tagged_label(key: str) -> str:
    name = key[len("tracked_") :] if key.startswith("tracked_") else key
    labels = {
        "energies": "Energy",
        "z2_order_abs": r"Z2 $|M|$",
        "z2_order2": r"Z2 $M^2$",
        "order_parameter": "Order parameter",
        "z2_order_parameter": "Z2 order parameter",
        "chirality_order_parameter": "Chirality order parameter",
        "helicities": "Helicity",
        "total_density": "Total density",
        "density": "Density",
        "amplitude_imbalance": "Amplitude imbalance",
    }
    return labels.get(name, name.replace("_", " ").title())


def _strip_tau_suffix(key: str) -> str:
    name = key
    for suffix in ["_tau_int_sweeps", "_tau_int"]:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _finite_stats(values: Any) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "min": np.nan,
            "max": np.nan,
            "sum": np.nan,
        }
    return {
        "n": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "sum": float(np.sum(finite)),
    }


def _selected_row_indices(
    n_rows: int,
    *,
    max_walkers_per_plot: int,
) -> np.ndarray:
    n_rows = int(n_rows)
    if n_rows <= 0:
        return np.empty(0, dtype=np.int64)
    max_walkers_per_plot = max(1, int(max_walkers_per_plot))
    if n_rows <= max_walkers_per_plot:
        return np.arange(n_rows, dtype=np.int64)
    selected = np.linspace(0, n_rows - 1, max_walkers_per_plot)
    return np.unique(np.rint(selected).astype(np.int64))


def _largest_L_with_tracked_key(
    analyzed_by_L: dict[int, dict[str, Any]],
    key: str,
) -> int | None:
    available: list[int] = []
    for L, obs in analyzed_by_L.items():
        values = np.asarray(obs.get(key))
        walkers = np.asarray(obs.get("tracked_walkers", []))
        if values.ndim == 1 and values.size > 0 and walkers.size == values.size:
            available.append(int(L))
    return max(available) if available else None


def _largest_L_with_tracked_histories(
    analyzed_by_L: dict[int, dict[str, Any]],
) -> int | None:
    available: list[int] = []
    for L, obs in analyzed_by_L.items():
        histories = obs.get("_tagged_autocorrelation_histories")
        walkers = np.asarray(obs.get("tracked_walkers", []))
        if isinstance(histories, dict) and histories and walkers.size > 0:
            available.append(int(L))
    return max(available) if available else None


def _has_temperature_curve(obs: dict[str, Any], key: str) -> bool:
    if key not in obs or "temps" not in obs:
        return False
    temps = np.asarray(obs["temps"])
    values = np.asarray(obs[key])
    return values.ndim == 1 and temps.ndim == 1 and values.shape == temps.shape


def _multi_L_colors(L_values: list[int]) -> dict[int, Any]:
    if not L_values:
        return {}
    cmap = plt.get_cmap("viridis")
    ordered = sorted(int(L) for L in L_values)
    denom = max(1, len(ordered) - 1)
    return {L: cmap(i / denom) for i, L in enumerate(ordered)}


def _multi_L_plot(
    ax: Any,
    x: Any,
    y: Any,
    *,
    L: int | None = None,
    color: Any = None,
    label: str | None = None,
    **kwargs: Any,
) -> Any:
    if label is None and L is not None:
        label = fr"$L={L}$"
    style = {
        "color": color,
        "markersize": MULTI_L_MARKERSIZE,
        "markeredgewidth": 0.0,
        "linewidth": MULTI_L_LINEWIDTH,
        "alpha": 0.95,
        "label": label,
    }
    style.update(kwargs)
    return ax.plot(x, y, "o-", **style)


def _format_axes(
    ax: Any,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    integer_x: bool = False,
) -> None:
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if integer_x:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.ticklabel_format(axis="both", style="plain", useOffset=False)
    ax.grid(alpha=0.28, linewidth=0.55)


def _legend(ax: Any) -> None:
    handles, labels = ax.get_legend_handles_labels()
    labels = [label for label in labels if not label.startswith("_")]
    if labels:
        ax.legend(fontsize=9)


def safe_filename_token(name: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name).strip().lower())
    return token.strip("._-") or "plot"


def finish_figure(fig: Any, out_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)
