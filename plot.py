#!/usr/bin/env python3
"""
Plot analyzed PTMC simulation outputs.
Usage
-----
From the project root:
    python plot.py outputs/amplitude_production
or simply:
    python plot.py amplitude_production
The script will write figures to:
    outputs/amplitude_production/plots/
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from analysis.autocorrelation_plots import (
    add_autocorrelation_aliases,
    attach_autocorrelation_metadata,
    infer_autocorrelation_keys,
    infer_autocorrelation_sample_strides,
    infer_tagged_autocorrelation_keys,
    infer_tagged_autocorrelation_sample_strides,
    summarize_autocorrelation,
)
from analysis.bkt import (
    estimate_bkt_intersections_by_L,
    estimate_weber_minnhagen_C,
    helicity_reference_label,
    helicity_reference_line,
)
from analysis.pipeline import analyze_run

RUN_FILE_GLOB = "*_L*.npz"
KEEP_FIGURES_OPEN = False
MULTI_L_MARKERSIZE = 2.0
MULTI_L_LINEWIDTH = 1.45
MULTI_L_ELINEWIDTH = 1.05
FIGSIZE = (7.0, 4.8)
FIG_DPI = 200

# ============================================================
# Loading helpers
# ============================================================

def _json_scalar_load(value: Any) -> Any:
    """
    Load a JSON object stored as a scalar NumPy value.
    """
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, (bytes, np.bytes_)):
        value = value.decode("utf-8")
    return json.loads(str(value))

def load_npz_as_dict(path: str | Path) -> dict[str, Any]:
    """
    Load one .npz file into a regular dictionary.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}

def load_params(data: dict[str, Any]) -> dict[str, Any]:
    """
    Load params_json if present.
    """
    if "params_json" not in data:
        return {}
    try:
        return _json_scalar_load(data["params_json"])
    except Exception:
        return {}

def resolve_output_dir(folder: str | Path) -> Path:
    """
    Accept either 'outputs/name' or just 'name'.
    """
    folder = Path(folder)
    if folder.exists():
        return folder
    candidate = Path("outputs") / folder
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"Could not find output folder {folder!s} or {candidate!s}."
    )

def find_run_files(output_dir: Path) -> list[Path]:
    """
    Find .npz run files in an output directory.
    """
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            files = [
                output_dir / filename
                for filename in manifest.get("files", [])
            ]
            files = [path for path in files if path.exists()]
            if files:
                return sorted(files)
        except Exception:
            pass
    return sorted(output_dir.glob(RUN_FILE_GLOB))

def get_L_from_data_or_filename(data: dict[str, Any], path: Path) -> int:
    """
    Get L from saved data, params_json, or filename.
    """
    if "L" in data:
        return int(np.asarray(data["L"]).item())
    params = load_params(data)
    if "L" in params:
        return int(params["L"])
    match = re.search(r"_L(\d+)", path.stem)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not determine L for {path}.")

# ============================================================
# Analysis-data normalization
# ============================================================

def add_legacy_aliases(data: dict[str, Any]) -> dict[str, Any]:
    """
    Add standard analysis keys from older/specific saved names.
    The analysis pipeline expects generic names like:

        order_abs_block_means
        order2_block_means
        order4_block_means
    Older amplitude outputs may use names like:
        mag_abs_block_means
        mag2_block_means
        mag4_block_means
    """
    out = dict(data)

    aliases = {
        "mag_abs_block_means": "order_abs_block_means",
        "mag2_block_means": "order2_block_means",
        "mag4_block_means": "order4_block_means",
    }
    for old_key, new_key in aliases.items():
        if new_key not in out and old_key in out:
            out[new_key] = out[old_key]

    return add_autocorrelation_aliases(out)

def infer_extra_observable_specs(data: dict[str, Any]) -> dict[str, dict[str, str]]:
    """
    Automatically detect extra scalar observables stored as block means.

    Example:
        nbar_block_means -> observable name 'nbar'
        total_amplitude_block_means -> observable name 'total_amplitude'
    """
    core_block_keys = {
        "energy_block_means",
        "energy2_block_means",
        "order_abs_block_means",
        "order2_block_means",
        "order4_block_means",
        "mag_abs_block_means",
        "mag2_block_means",
        "mag4_block_means",
        "helicity_Kx_block_means",
        "helicity_Ix_block_means",
        "helicity_Ix2_block_means",
        "helicity_Ky_block_means",
        "helicity_Iy_block_means",
        "helicity_Iy2_block_means",
    }
    specs: dict[str, dict[str, str]] = {}
    for key in data:
        if not key.endswith("_block_means"):
            continue
        if key in core_block_keys:
            continue
        if key.startswith("helicity_"):
            continue
        name = key[: -len("_block_means")]
        if name.endswith("2") or name.endswith("4"):
            continue
        specs[name] = {"block_key": key}
    return specs

def analyze_output_folder(
    output_dir: Path,
    *,
    weber_minnhagen_C: float | None = None,
    bkt_n_bootstrap: int = 2000,
    bkt_rng_seed: int = 12345,
) -> dict[int, dict[str, Any]]:
    """
    Load and analyze every L run in an output folder.
    """
    run_files = find_run_files(output_dir)
    if not run_files:
        raise FileNotFoundError(f"No run files matching {RUN_FILE_GLOB!r} found.")
    analyzed_by_L: dict[int, dict[str, Any]] = {}
    for path in run_files:
        raw_data = load_npz_as_dict(path)
        data = add_legacy_aliases(raw_data)
        params = load_params(data)
        L = get_L_from_data_or_filename(data, path)
        record_stride = int(params.get("record_stride", 1))
        energy_per_site = bool(params.get("energy_per_site", False))
        order_parameter_per_site = bool(
            params.get("order_parameter_per_site", False)
        )
        extra_specs = infer_extra_observable_specs(data)
        autocorr_keys = infer_autocorrelation_keys(data)
        autocorr_strides = infer_autocorrelation_sample_strides(
            data,
            autocorr_keys,
            params,
        )
        tagged_autocorr_keys = infer_tagged_autocorrelation_keys(data)
        tagged_autocorr_strides = infer_tagged_autocorrelation_sample_strides(
            data,
            tagged_autocorr_keys,
            params,
        )
        obs = analyze_run(
            data,
            L=L,
            temps=data.get("temps"),
            energy_per_site=energy_per_site,
            order_parameter_per_site=order_parameter_per_site,
            record_stride=record_stride,
            autocorrelation_keys=autocorr_keys,
            autocorrelation_sample_strides=autocorr_strides,
            tagged_autocorrelation_keys=tagged_autocorr_keys,
            tagged_autocorrelation_sample_strides=tagged_autocorr_strides,
            extra_observable_specs=extra_specs,
            weber_minnhagen_C=weber_minnhagen_C,
            bkt_n_bootstrap=bkt_n_bootstrap,
            bkt_rng_seed=bkt_rng_seed + int(L),
        )
        obs["_source_file"] = str(path)
        obs["_params"] = params
        obs["_extra_observable_names"] = sorted(extra_specs.keys())
        attach_autocorrelation_metadata(
            obs,
            data,
            autocorrelation_keys=autocorr_keys,
            autocorrelation_sample_strides=autocorr_strides,
            tagged_autocorrelation_keys=tagged_autocorr_keys,
            tagged_autocorrelation_sample_strides=tagged_autocorr_strides,
        )
        if "label_positions" in data:
            obs["label_positions"] = data["label_positions"]
            obs["label_position_record_stride"] = np.int32(record_stride)
        analyzed_by_L[L] = obs
    return dict(sorted(analyzed_by_L.items()))


def _finite_float_or_none(value: Any) -> float | None:
    """
    Convert a candidate numeric value to float, returning None for non-finite.
    """
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def auto_select_weber_minnhagen_C(
    analyzed_by_L: dict[int, dict[str, Any]],
    *,
    provided_C: float | None = None,
    fit_enabled: bool = True,
    scan_lo: float | None = None,
    scan_hi: float | None = None,
    scan_points: int = 5000,
    min_sizes: int = 3,
    require_all_sizes: bool = True,
) -> tuple[float | None, dict[str, Any]]:
    """
    Choose the Weber-Minnhagen constant used for BKT plotting/crossings.

    Priority:
        1. explicit --weber-minnhagen-C, if supplied;
        2. automatic multi-L fit, if possible;
        3. None, which means the bare 2T/pi line.
    """
    explicit_C = _finite_float_or_none(provided_C)
    if explicit_C is not None:
        return explicit_C, {
            "available": True,
            "source": "user supplied",
            "C": explicit_C,
            "C_err": np.nan,
        }

    if not fit_enabled:
        return None, {
            "available": False,
            "source": "disabled",
            "reason": "automatic Weber-Minnhagen C fit disabled",
        }

    fit = estimate_weber_minnhagen_C(
        analyzed_by_L,
        min_sizes=min_sizes,
        scan_lo=scan_lo,
        scan_hi=scan_hi,
        scan_points=scan_points,
        require_all_sizes=require_all_sizes,
    )
    fit = dict(fit)
    fit["source"] = "automatic fit"

    if fit.get("available", False):
        C = _finite_float_or_none(fit.get("C"))
        if C is not None:
            return C, fit

    return None, fit


def update_helicity_intersections_with_C(
    analyzed_by_L: dict[int, dict[str, Any]],
    *,
    weber_minnhagen_C: float | None,
    bkt_n_bootstrap: int = 2000,
    bkt_rng_seed: int = 12345,
) -> None:
    """
    Recompute per-L helicity/reference intersections after choosing C.
    """
    estimates = estimate_bkt_intersections_by_L(
        analyzed_by_L,
        weber_minnhagen_C=weber_minnhagen_C,
        n_bootstrap=bkt_n_bootstrap,
        rng_seed=bkt_rng_seed,
    )
    for L, estimate in estimates.items():
        if int(L) in analyzed_by_L:
            analyzed_by_L[int(L)]["bkt_intersection"] = estimate


def attach_weber_minnhagen_fit(
    analyzed_by_L: dict[int, dict[str, Any]],
    fit: dict[str, Any],
) -> None:
    """
    Store the global Weber-Minnhagen fit result on each analyzed run.
    """
    for obs in analyzed_by_L.values():
        obs["weber_minnhagen_fit"] = fit


# ============================================================
# Plot helpers
# ============================================================

def configure_plot_style() -> None:
    """
    Apply compact, publication-style Matplotlib defaults.
    """
    plt.rcParams.update(
        {
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linewidth": 0.55,
            "legend.frameon": False,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )

def _multi_L_colors(keys: Any) -> dict[int, Any]:
    """
    Return rainbow colors keyed by sorted system size.
    """
    sorted_keys = sorted(int(key) for key in keys)
    if not sorted_keys:
        return {}
    cmap = plt.get_cmap("rainbow")
    positions = np.linspace(0.02, 0.98, max(len(sorted_keys), 2))
    return {
        key: cmap(positions[i])
        for i, key in enumerate(sorted_keys)
    }

def _multi_L_errorbar(
    ax: Any,
    x: Any,
    y: Any,
    yerr: Any = None,
    *,
    L: int | None = None,
    color: Any = None,
    label: str | None = None,
    **kwargs: Any,
) -> Any:
    """
    Reference-style all-L errorbar helper.
    """
    if label is None and L is not None:
        label = fr"$L={L}$"
    style = {
        "fmt": "o-",
        "color": color,
        "ecolor": color,
        "markersize": MULTI_L_MARKERSIZE,
        "markeredgewidth": 0.0,
        "linewidth": MULTI_L_LINEWIDTH,
        "elinewidth": MULTI_L_ELINEWIDTH,
        "capsize": 0,
        "alpha": 0.95,
        "label": label,
    }
    style.update(kwargs)
    return ax.errorbar(x, y, yerr=yerr, **style)

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
    """
    Reference-style all-L line helper.
    """
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
    """
    Apply common axes labels and light styling.
    """
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
    """
    Add a compact legend if any labeled artists exist.
    """
    handles, labels = ax.get_legend_handles_labels()
    labels = [label for label in labels if not label.startswith("_")]
    if labels:
        ax.legend(fontsize=9)

def safe_filename_token(name: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name).strip().lower())
    return token.strip("._-") or "plot"

def has_temperature_curve(obs: dict[str, Any], key: str) -> bool:
    if key not in obs or "temps" not in obs:
        return False
    temps = np.asarray(obs["temps"])
    values = np.asarray(obs[key])
    return values.ndim == 1 and temps.ndim == 1 and values.shape == temps.shape

def _finite_curve(
    temps: Any,
    values: Any,
    errors: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return finite x/y/error arrays, using zero errors if none are supplied.
    """
    temps = np.asarray(temps, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if errors is None:
        errors = np.zeros_like(values)
    errors = np.asarray(errors, dtype=np.float64)
    if errors.shape != values.shape:
        errors = np.zeros_like(values)
    mask = np.isfinite(temps) & np.isfinite(values)
    errors = np.where(np.isfinite(errors), np.maximum(errors, 0.0), 0.0)
    return temps[mask], values[mask], errors[mask]

def _sorted_unique_curve(
    temps: Any,
    values: Any,
    errors: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sort a finite curve by temperature and keep one value per temperature.
    """
    temps = np.asarray(temps, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    errors = np.asarray(errors, dtype=np.float64)
    order = np.argsort(temps)
    temps = temps[order]
    values = values[order]
    errors = errors[order]
    temps, unique_idx = np.unique(temps, return_index=True)
    return temps, values[unique_idx], errors[unique_idx]

def _common_temperature_grid(temps_a: Any, temps_b: Any) -> np.ndarray:
    """
    Build a shared interpolation grid for two temperature curves.
    """
    temps_a = np.asarray(temps_a, dtype=np.float64)
    temps_b = np.asarray(temps_b, dtype=np.float64)
    lo = max(float(np.min(temps_a)), float(np.min(temps_b)))
    hi = min(float(np.max(temps_a)), float(np.max(temps_b)))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        return np.empty(0, dtype=np.float64)
    grid = np.unique(np.concatenate((temps_a, temps_b, np.array([lo, hi]))))
    return grid[(grid >= lo) & (grid <= hi)]

def _piecewise_linear_roots(temps: Any, diff: Any) -> list[tuple[float, float, int]]:
    """
    Find all sign-changing roots of a piecewise-linear curve.
    """
    temps = np.asarray(temps, dtype=np.float64)
    diff = np.asarray(diff, dtype=np.float64)
    finite = np.isfinite(temps) & np.isfinite(diff)
    temps = temps[finite]
    diff = diff[finite]
    if temps.size < 2:
        return []
    order = np.argsort(temps)
    temps = temps[order]
    diff = diff[order]
    candidates: list[tuple[float, float, int]] = []
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

def _curve_curve_crossing_candidates(
    temps_a: Any,
    values_a: Any,
    temps_b: Any,
    values_b: Any,
) -> list[tuple[float, float, int]]:
    """
    Return piecewise-linear crossing candidates between two curves.
    """
    grid = _common_temperature_grid(temps_a, temps_b)
    if grid.size < 2:
        return []
    diff = np.interp(grid, temps_a, values_a) - np.interp(
        grid,
        temps_b,
        values_b,
    )
    return _piecewise_linear_roots(grid, diff)

def _curve_curve_crossing_temperature(
    temps_a: Any,
    values_a: Any,
    temps_b: Any,
    values_b: Any,
    *,
    target: float | None = None,
) -> float | None:
    """
    Pick one curve-curve crossing, optionally nearest a target temperature.
    """
    candidates = _curve_curve_crossing_candidates(
        temps_a,
        values_a,
        temps_b,
        values_b,
    )
    if not candidates:
        return None
    if target is not None and np.isfinite(target):
        root = min(candidates, key=lambda item: abs(item[0] - float(target)))
    else:
        root = min(candidates, key=lambda item: item[1])
    return float(root[0])

def _peak_temperature(obs: dict[str, Any], key: str) -> tuple[float, float] | None:
    """
    Return (T, value) for the largest finite value of a temperature curve.
    """
    if not has_temperature_curve(obs, key):
        return None
    temps = np.asarray(obs["temps"], dtype=np.float64)
    values = np.asarray(obs[key], dtype=np.float64)
    finite = np.isfinite(temps) & np.isfinite(values)
    if not np.any(finite):
        return None
    temps = temps[finite]
    values = values[finite]
    i = int(np.argmax(values))
    return float(temps[i]), float(values[i])

def _bootstrap_error(central: float, roots: list[float]) -> tuple[float, int]:
    """
    Compute a robust bootstrap spread for crossing temperatures.
    """
    roots_arr = np.asarray(roots, dtype=np.float64)
    roots_arr = roots_arr[np.isfinite(roots_arr)]
    if roots_arr.size < 2:
        return np.nan, int(roots_arr.size)
    spread = np.abs(roots_arr - float(central))
    keep = spread <= np.nanpercentile(spread, 95.0)
    kept = roots_arr[keep]
    if kept.size < 2:
        kept = roots_arr
    return float(np.std(kept, ddof=1)), int(kept.size)

def estimate_largest_L_binder_crossing(
    analyzed_by_L: dict[int, dict[str, Any]],
    *,
    crossing_target: float | None = None,
    n_bootstrap: int = 2000,
    rng_seed: int = 23456,
) -> dict[str, Any]:
    """
    Estimate the Binder crossing between the two largest available sizes.
    """
    sizes = [
        int(L)
        for L, obs in analyzed_by_L.items()
        if has_temperature_curve(obs, "U4")
    ]
    sizes = sorted(sizes)
    if len(sizes) < 2:
        return {
            "available": False,
            "reason": "need at least two system sizes with Binder data",
        }

    L_a, L_b = sizes[-2], sizes[-1]
    obs_a = analyzed_by_L[L_a]
    obs_b = analyzed_by_L[L_b]
    temps_a, values_a, errors_a = _finite_curve(
        obs_a["temps"],
        obs_a["U4"],
        obs_a.get("U4_err"),
    )
    temps_b, values_b, errors_b = _finite_curve(
        obs_b["temps"],
        obs_b["U4"],
        obs_b.get("U4_err"),
    )
    if temps_a.size < 2 or temps_b.size < 2:
        return {
            "available": False,
            "L_pair": [L_a, L_b],
            "reason": "need at least two finite Binder points for each size",
        }

    temps_a, values_a, errors_a = _sorted_unique_curve(
        temps_a,
        values_a,
        errors_a,
    )
    temps_b, values_b, errors_b = _sorted_unique_curve(
        temps_b,
        values_b,
        errors_b,
    )
    candidates = _curve_curve_crossing_candidates(
        temps_a,
        values_a,
        temps_b,
        values_b,
    )
    if not candidates:
        return {
            "available": False,
            "L_pair": [L_a, L_b],
            "reason": "no sign-changing Binder crossing in the shared T range",
        }

    target_source = "user supplied"
    selection_target = crossing_target
    if selection_target is None or not np.isfinite(selection_target):
        peak = _peak_temperature(obs_b, "chi") or _peak_temperature(obs_b, "C")
        if peak is not None:
            selection_target = peak[0]
            target_source = "largest-L finite-size peak"
        else:
            target_source = "closest-approach fallback"

    if selection_target is not None and np.isfinite(selection_target):
        selected = min(candidates, key=lambda item: abs(item[0] - selection_target))
    else:
        selected = min(candidates, key=lambda item: item[1])
    T_cross = float(selected[0])

    rng = np.random.default_rng(rng_seed)
    roots: list[float] = []
    for _ in range(max(0, int(n_bootstrap))):
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
    value_a = float(np.interp(T_cross, temps_a, values_a))
    value_b = float(np.interp(T_cross, temps_b, values_b))

    return {
        "available": True,
        "L_pair": [L_a, L_b],
        "T": T_cross,
        "T_err": T_err,
        "value": 0.5 * (value_a + value_b),
        "target": (
            float(selection_target)
            if selection_target is not None and np.isfinite(selection_target)
            else np.nan
        ),
        "target_source": target_source,
        "crossing_temperatures": [
            float(candidate[0])
            for candidate in candidates
        ],
        "n_bootstrap": n_used,
    }

def finish_figure(fig: Any, out_path: Path, *, tight_layout: bool = True) -> None:
    """
    Save a figure and close it unless interactive display was requested.
    """
    if tight_layout:
        fig.tight_layout()
        fig.savefig(out_path, dpi=FIG_DPI)
    else:
        fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    if not KEEP_FIGURES_OPEN:
        plt.close(fig)

def plot_temperature_curve(
    analyzed_by_L: dict[int, dict[str, Any]],
    *,
    key: str,
    err_key: str | None,
    ylabel: str,
    title: str,
    out_path: Path,
) -> bool:
    """
    Plot one observable versus temperature for all available L.
    """
    available = [
        L for L, obs in analyzed_by_L.items()
        if has_temperature_curve(obs, key)
    ]
    if not available:
        return False
    colors = _multi_L_colors(available)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for L in available:
        obs = analyzed_by_L[L]
        temps = np.asarray(obs["temps"], dtype=np.float64)
        values = np.asarray(obs[key], dtype=np.float64)
        errors = None
        if err_key is not None and err_key in obs:
            maybe_errors = np.asarray(obs[err_key], dtype=np.float64)
            if maybe_errors.shape == values.shape:
                errors = maybe_errors
        if errors is None or not np.any(np.isfinite(errors)):
            _multi_L_plot(
                ax,
                temps,
                values,
                L=L,
                color=colors.get(int(L)),
            )
        else:
            _multi_L_errorbar(
                ax,
                temps,
                values,
                yerr=errors,
                L=L,
                color=colors.get(int(L)),
            )
    _format_axes(
        ax,
        xlabel=r"Temperature $T$",
        ylabel=ylabel,
        title=title,
    )
    _legend(ax)
    finish_figure(fig, out_path)
    return True

def plot_index_curve(
    analyzed_by_L: dict[int, dict[str, Any]],
    *,
    key: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> bool:
    """
    Plot a one-dimensional diagnostic versus index for all available L.
    """
    available = []
    for L, obs in analyzed_by_L.items():
        if key not in obs:
            continue
        arr = np.asarray(obs[key])
        if arr.ndim == 1 and arr.size > 0:
            available.append(L)
    if not available:
        return False
    colors = _multi_L_colors(available)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for L in available:
        values = np.asarray(analyzed_by_L[L][key], dtype=np.float64)
        x = np.arange(values.size)

        _multi_L_plot(
            ax,
            x,
            values,
            L=L,
            color=colors.get(int(L)),
        )
    _format_axes(
        ax,
        xlabel="Index",
        ylabel=ylabel,
        title=title,
        integer_x=True,
    )
    _legend(ax)
    finish_figure(fig, out_path)
    return True

# ============================================================
# Standard observable plots
# ============================================================

STANDARD_TEMPERATURE_PLOTS = [
    ("e", "e_err", r"Energy per site $e$", "Energy per site", "energy_per_site"),
    ("C", "C_err", r"Specific heat $C$", "Specific heat", "specific_heat"),
    (
        "m_abs",
        "m_abs_err",
        r"Z$_2$ order parameter $\langle |m| \rangle$",
        "Z2 order parameter",
        "z2_order_parameter",
    ),
    ("chi", "chi_err", r"Susceptibility $\chi$", "Susceptibility", "susceptibility"),
    ("U4", "U4_err", r"Binder cumulant $U_4$", "Binder cumulant", "binder_cumulant"),
    (
        "binder_ratio",
        "binder_ratio_err",
        r"Binder ratio $\langle m^4\rangle / 3\langle m^2\rangle^2$",
        "Binder ratio",
        "binder_ratio",
    ),
]

def plot_standard_temperature_observables(
    analyzed_by_L: dict[int, dict[str, Any]],
    plots_dir: Path,
) -> list[Path]:
    """
    Plot standard thermodynamic, Z2, and helicity observables.
    """
    written: list[Path] = []
    for key, err_key, ylabel, title, filename in STANDARD_TEMPERATURE_PLOTS:
        out_path = plots_dir / f"{filename}.png"
        ok = plot_temperature_curve(
            analyzed_by_L,
            key=key,
            err_key=err_key,
            ylabel=ylabel,
            title=title,
            out_path=out_path,
        )
        if ok:
            written.append(out_path)
    return written


def plot_helicity_bkt_reference(
    analyzed_by_L: dict[int, dict[str, Any]],
    plots_dir: Path,
    *,
    weber_minnhagen_C: float | None = None,
) -> list[Path]:
    """
    Plot helicity modulus with the bare BKT reference line.

    If a Weber-Minnhagen C is available, scale each measured finite-L
    helicity curve by the corresponding finite-size factor instead of scaling
    the reference line:

        Y_scaled = Y_L / [1 + 1 / (2 log L + C)].

    Crossings are reported in the terminal summary; no vertical crossing
    lines are drawn here.
    """
    available = [
        L for L, obs in analyzed_by_L.items()
        if has_temperature_curve(obs, "Y")
    ]
    if not available:
        return []

    colors = _multi_L_colors(available)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    reference_L = max(int(L) for L in available)
    use_wm_scaling = (
        weber_minnhagen_C is not None and np.isfinite(weber_minnhagen_C)
    )

    for L in available:
        obs = analyzed_by_L[L]
        temps = np.asarray(obs["temps"], dtype=np.float64)
        Y = np.asarray(obs["Y"], dtype=np.float64)
        Y_err = np.asarray(obs.get("Y_err", np.full_like(Y, np.nan)))
        color = colors.get(int(L))
        if use_wm_scaling:
            denominator = 2.0 * np.log(float(L)) + float(weber_minnhagen_C)
            if np.isfinite(denominator) and denominator > 0.0:
                scale = 1.0 + 1.0 / denominator
                Y = Y / scale
                Y_err = Y_err / scale

        if Y_err.shape == Y.shape and np.any(np.isfinite(Y_err)):
            _multi_L_errorbar(
                ax,
                temps,
                Y,
                yerr=Y_err,
                L=L,
                color=color,
            )
        else:
            _multi_L_plot(
                ax,
                temps,
                Y,
                L=L,
                color=color,
            )

        if int(L) == reference_L:
            reference = helicity_reference_line(
                temps,
            )
            ax.plot(
                temps,
                reference,
                "--",
                color="0.25",
                linewidth=1.25,
                alpha=0.85,
                label=fr"{helicity_reference_label(None)}",
            )

    ylabel = r"Helicity modulus $Y$"
    title = "Helicity modulus with BKT reference"
    if use_wm_scaling:
        ylabel = r"Scaled helicity $Y/[1+1/(2\log L+C)]$"
        title = "Weber-Minnhagen scaled helicity with bare BKT reference"

    _format_axes(
        ax,
        xlabel=r"Temperature $T$",
        ylabel=ylabel,
        title=title,
    )
    _legend(ax)

    out_path = plots_dir / "helicity_modulus.png"
    finish_figure(fig, out_path)
    return [out_path]


def plot_helicity_diagnostics(
    analyzed_by_L: dict[int, dict[str, Any]],
    plots_dir: Path,
) -> list[Path]:
    """
    Plot helicity component diagnostics in one multi-panel figure.

    This replaces separate plots for Kx, Ky, Ix, Iy, Ix2, and Iy2.
    """
    panels = [
        ("helicity_Kx", "helicity_Kx_err", r"$K_x$"),
        ("helicity_Ky", "helicity_Ky_err", r"$K_y$"),
        ("helicity_Ix", "helicity_Ix_err", r"$I_x$"),
        ("helicity_Iy", "helicity_Iy_err", r"$I_y$"),
        ("helicity_Ix2", "helicity_Ix2_err", r"$I_x^2$"),
        ("helicity_Iy2", "helicity_Iy2_err", r"$I_y^2$"),
    ]

    available_L = sorted(
        {
            int(L)
            for L, obs in analyzed_by_L.items()
            if "temps" in obs
            and any(has_temperature_curve(obs, key) for key, _, _ in panels)
        }
    )
    if not available_L:
        return []

    colors = _multi_L_colors(available_L)
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.6), sharex=True)
    flat_axes = axes.ravel()

    for ax, (key, err_key, ylabel) in zip(flat_axes, panels):
        for L in available_L:
            obs = analyzed_by_L[L]
            if not has_temperature_curve(obs, key):
                continue

            temps = np.asarray(obs["temps"], dtype=np.float64)
            values = np.asarray(obs[key], dtype=np.float64)
            color = colors.get(int(L))

            errors = None
            if err_key in obs:
                maybe_errors = np.asarray(obs[err_key], dtype=np.float64)
                if maybe_errors.shape == values.shape and np.any(np.isfinite(maybe_errors)):
                    errors = maybe_errors

            if errors is None:
                _multi_L_plot(
                    ax,
                    temps,
                    values,
                    L=L,
                    color=color,
                    label=fr"$L={L}$",
                )
            else:
                _multi_L_errorbar(
                    ax,
                    temps,
                    values,
                    yerr=errors,
                    L=L,
                    color=color,
                    label=fr"$L={L}$",
                )

        _format_axes(
            ax,
            xlabel=r"Temperature $T$",
            ylabel=ylabel,
            title=ylabel,
        )

    handles, labels = flat_axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(len(labels), 4),
            frameon=False,
        )

    fig.suptitle("Helicity diagnostics", y=0.995)
    out_path = plots_dir / "helicity_diagnostics.png"
    finish_figure(fig, out_path)
    return [out_path]

def plot_extra_observables(
    analyzed_by_L: dict[int, dict[str, Any]],
    plots_dir: Path,
) -> list[Path]:
    """
    Plot automatically detected extra block observables.
    """
    names: set[str] = set()
    for obs in analyzed_by_L.values():
        names.update(obs.get("_extra_observable_names", []))

    written: list[Path] = []
    for name in sorted(names):
        out_path = plots_dir / f"{safe_filename_token(name)}.png"
        ok = plot_temperature_curve(
            analyzed_by_L,
            key=name,
            err_key=f"{name}_err",
            ylabel=name.replace("_", " "),
            title=name.replace("_", " ").title(),
            out_path=out_path,
        )
        if ok:
            written.append(out_path)
    return written

# ============================================================
# Summary
# ============================================================

def _json_clean(value: Any) -> Any:
    """
    Convert NumPy values and non finite floats to JSON friendly objects.
    """
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_clean(value.tolist())
    if isinstance(value, np.generic):
        return _json_clean(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value

def _finite_stats(values: Any) -> dict[str, Any]:
    """
    Return compact finite statistics for a numeric array.
    """
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

def _peak_record(obs: dict[str, Any], key: str) -> dict[str, float] | None:
    """
    Return a JSON-ready peak record for a temperature observable.
    """
    peak = _peak_temperature(obs, key)
    if peak is None:
        return None
    return {
        "T": peak[0],
        "value": peak[1],
    }

def build_analysis_report(
    analyzed_by_L: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """
    Build the printed/JSON summary of critical and diagnostic estimates.
    """
    binder_crossing = estimate_largest_L_binder_crossing(analyzed_by_L)
    weber_minnhagen_fit: dict[str, Any] = {}
    for obs in analyzed_by_L.values():
        candidate = obs.get("weber_minnhagen_fit")
        if isinstance(candidate, dict):
            weber_minnhagen_fit = candidate
            break
    runs: dict[str, Any] = {}
    for L, obs in analyzed_by_L.items():
        run: dict[str, Any] = {}
        C_peak = _peak_record(obs, "C")
        chi_peak = _peak_record(obs, "chi")
        if C_peak is not None:
            run["specific_heat_peak"] = C_peak
        if chi_peak is not None:
            run["susceptibility_peak"] = chi_peak

        bkt = obs.get("bkt_intersection")
        if isinstance(bkt, dict):
            run["bkt_intersection"] = bkt

        if "swap_rate_mean" in obs:
            run["swap_acceptance"] = {
                "mean": obs.get("swap_rate_mean"),
                "min": obs.get("swap_rate_min"),
                "max": obs.get("swap_rate_max"),
                "accepted_total": obs.get("swap_acceptance_total"),
                "attempts_total": obs.get("swap_attempts_total"),
            }

        local_acceptance: dict[str, Any] = {}
        suffix = "_acceptance_rate_mean"
        for key in sorted(obs):
            if not key.endswith(suffix):
                continue
            name = key[: -len(suffix)]
            local_acceptance[name] = {
                "mean": obs.get(f"{name}_acceptance_rate_mean"),
                "min": obs.get(f"{name}_acceptance_rate_min"),
                "max": obs.get(f"{name}_acceptance_rate_max"),
                "accepted_total": obs.get(f"{name}_accepted_total"),
                "attempted_total": obs.get(f"{name}_attempted_total"),
            }
        if local_acceptance:
            run["local_acceptance"] = local_acceptance

        if "round_trip_counts" in obs:
            counts = np.asarray(obs["round_trip_counts"])
            run["round_trips"] = {
                "total": int(np.sum(counts)),
                "walkers_with_round_trips": int(np.count_nonzero(counts)),
                "max_per_walker": int(np.max(counts)) if counts.size else 0,
            }
        if "commute_counts" in obs:
            counts = np.asarray(obs["commute_counts"])
            run["commutes"] = {
                "total": int(np.sum(counts)),
                "walkers_with_commutes": int(np.count_nonzero(counts)),
                "max_per_walker": int(np.max(counts)) if counts.size else 0,
            }
        if "round_trip_durations" in obs:
            run["round_trip_duration"] = _finite_stats(
                obs["round_trip_durations"]
            )
        if "hit_both_edges" in obs:
            hits = np.asarray(obs["hit_both_edges"], dtype=np.bool_)
            run["walkers_hit_both_edges"] = {
                "count": int(np.count_nonzero(hits)),
                "total": int(hits.size),
            }

        energy_drift: dict[str, Any] = {}
        for key in [
            "energy_drift_abs_max",
            "energy_drift_abs_max_global",
            "energy_drift_recompute_count_total",
        ]:
            if key in obs:
                energy_drift[key] = obs[key]
        if energy_drift:
            run["energy_drift"] = energy_drift

        run.update(summarize_autocorrelation(obs))

        runs[str(int(L))] = run

    return {
        "binder_crossing": binder_crossing,
        "weber_minnhagen_fit": weber_minnhagen_fit,
        "runs": runs,
    }

def _fmt_float(value: Any, *, digits: int = 5) -> str:
    """
    Format a number for terminal summaries.
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}g}"

def print_analysis_report(
    *,
    output_dir: Path,
    plots_dir: Path,
    analyzed_by_L: dict[int, dict[str, Any]],
    written: list[Path],
    summary_path: Path,
    report: dict[str, Any],
) -> None:
    """
    Print a compact, reference-style terminal summary.
    """
    L_values = sorted(int(L) for L in analyzed_by_L)
    print(f"Plotted output folder: {output_dir}")
    print(f"L values: {', '.join(str(L) for L in L_values)}")

    binder = report.get("binder_crossing", {})
    if isinstance(binder, dict) and binder.get("available", False):
        L_a, L_b = binder.get("L_pair", ["?", "?"])
        print(
            "Binder crossing "
            f"L={L_a}/{L_b}: "
            f"T={_fmt_float(binder.get('T'))} "
            f"+/- {_fmt_float(binder.get('T_err'), digits=2)}, "
            f"U4={_fmt_float(binder.get('value'))}"
        )
    elif isinstance(binder, dict):
        print(f"Binder crossing: unavailable ({binder.get('reason', 'no reason')})")

    wm_fit = report.get("weber_minnhagen_fit", {})
    if isinstance(wm_fit, dict) and wm_fit.get("available", False):
        source = wm_fit.get("source", "automatic fit")
        print(
            "Weber-Minnhagen C "
            f"({source}): "
            f"C={_fmt_float(wm_fit.get('C'))} "
            f"+/- {_fmt_float(wm_fit.get('C_err'), digits=2)}, "
            f"T_fit={_fmt_float(wm_fit.get('T'))}, "
            f"chi2/dof={_fmt_float(wm_fit.get('red_chi2'), digits=3)}"
        )
    elif isinstance(wm_fit, dict) and wm_fit:
        print(
            "Weber-Minnhagen C: unavailable "
            f"({wm_fit.get('reason', 'no reason')}); using bare BKT line"
        )

    printed_helicity = False
    for L in L_values:
        run = report.get("runs", {}).get(str(L), {})
        bkt = run.get("bkt_intersection")
        if not isinstance(bkt, dict) or not bkt.get("available", False):
            continue
        printed_helicity = True
        wm_C = bkt.get("weber_minnhagen_C")
        uses_wm = wm_C is not None and np.isfinite(float(wm_C))
        line_name = "Weber-Minnhagen" if uses_wm else "bare BKT"
        print(
            f"Helicity crossing L={L} ({line_name}): "
            f"T={_fmt_float(bkt.get('T'))} "
            f"+/- {_fmt_float(bkt.get('T_err'), digits=2)}"
        )
    if not printed_helicity:
        print("Helicity crossing: unavailable")

    for L in L_values:
        run = report.get("runs", {}).get(str(L), {})
        parts = []
        C_peak = run.get("specific_heat_peak")
        if isinstance(C_peak, dict):
            parts.append(
                "C peak "
                f"T={_fmt_float(C_peak.get('T'))}, "
                f"C={_fmt_float(C_peak.get('value'))}"
            )
        chi_peak = run.get("susceptibility_peak")
        if isinstance(chi_peak, dict):
            parts.append(
                "chi peak "
                f"T={_fmt_float(chi_peak.get('T'))}, "
                f"chi={_fmt_float(chi_peak.get('value'))}"
            )
        swap = run.get("swap_acceptance")
        if isinstance(swap, dict):
            parts.append(
                "swap "
                f"mean={_fmt_float(swap.get('mean'))}, "
                f"min={_fmt_float(swap.get('min'))}"
            )
        round_trips = run.get("round_trips")
        if isinstance(round_trips, dict):
            parts.append(f"round trips={round_trips.get('total', 0)}")
        if parts:
            print(f"L={L}: " + "; ".join(parts))

    print(f"Wrote {len(written)} plot files to: {plots_dir}")
    print(f"Wrote summary: {summary_path}")

def write_plot_summary(
    *,
    output_dir: Path,
    plots_dir: Path,
    analyzed_by_L: dict[int, dict[str, Any]],
    written: list[Path],
    analysis_report: dict[str, Any],
) -> Path:
    """
    Write a small JSON summary of what was plotted.
    """
    summary = {
        "output_dir": str(output_dir),
        "plots_dir": str(plots_dir),
        "L_values": sorted(int(L) for L in analyzed_by_L),
        "n_plots": len(written),
        "plots": [path.name for path in written],
        "analysis": analysis_report,
        "runs": {
            str(L): {
                "source_file": obs.get("_source_file"),
                "extra_observables": obs.get("_extra_observable_names", []),
                "autocorrelation_keys": obs.get("_autocorrelation_keys", []),
                "autocorrelation_sample_strides": obs.get(
                    "_autocorrelation_sample_strides",
                    {},
                ),
                "tagged_autocorrelation_keys": obs.get(
                    "_tagged_autocorrelation_keys",
                    [],
                ),
                "tagged_autocorrelation_sample_strides": obs.get(
                    "_tagged_autocorrelation_sample_strides",
                    {},
                ),
                "bkt_intersection": obs.get("bkt_intersection"),
            }
            for L, obs in analyzed_by_L.items()
        },
    }
    out_path = plots_dir / "plot_summary.json"
    out_path.write_text(
        json.dumps(_json_clean(summary), indent=2),
        encoding="utf-8",
    )
    return out_path

# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot PTMC output folder.",
    )
    parser.add_argument(
        "output_folder",
        help=(
            "Output folder to plot. Accepts either 'outputs/name' "
            "or just 'name'."
        ),
    )
    parser.add_argument(
        "--plots-dir",
        default=None,
        help=(
            "Directory for plots. Defaults to '<output_folder>/plots'."
        ),
    )
    parser.add_argument(
        "--weber-minnhagen-C",
        type=float,
        default=None,
        help=(
            "Override the automatic Weber-Minnhagen C fit and use this "
            "fixed C in (2T/pi)[1 + 1/(2 log L + C)]."
        ),
    )
    parser.add_argument(
        "--no-fit-weber-minnhagen-C",
        action="store_true",
        help=(
            "Disable automatic Weber-Minnhagen C fitting. If no explicit "
            "--weber-minnhagen-C is supplied, use the bare 2T/pi line."
        ),
    )
    parser.add_argument(
        "--wm-scan-lo",
        type=float,
        default=None,
        help="Optional lower temperature limit for automatic C fitting.",
    )
    parser.add_argument(
        "--wm-scan-hi",
        type=float,
        default=None,
        help="Optional upper temperature limit for automatic C fitting.",
    )
    parser.add_argument(
        "--wm-scan-points",
        type=int,
        default=5000,
        help="Number of candidate temperatures sampled for automatic C fitting.",
    )
    parser.add_argument(
        "--wm-min-sizes",
        type=int,
        default=3,
        help="Minimum number of system sizes required for automatic C fitting.",
    )
    parser.add_argument(
        "--wm-allow-missing-sizes",
        action="store_true",
        help=(
            "Allow a candidate fit temperature to use a subset of available "
            "sizes, as long as --wm-min-sizes is satisfied. By default all "
            "sizes must be usable at the chosen temperature."
        ),
    )
    parser.add_argument(
        "--bkt-bootstrap",
        type=int,
        default=2000,
        help="Number of bootstrap samples for helicity/BKT crossing errors.",
    )
    parser.add_argument(
        "--bkt-rng-seed",
        type=int,
        default=12345,
        help="Random seed for helicity/BKT crossing bootstrap errors.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively after saving.",
    )
    return parser.parse_args()


def main() -> None:
    global KEEP_FIGURES_OPEN
    args = parse_args()
    KEEP_FIGURES_OPEN = bool(args.show)
    configure_plot_style()
    output_dir = resolve_output_dir(args.output_folder)
    if args.plots_dir is None:
        plots_dir = output_dir / "plots"
    else:
        plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    analyzed_by_L = analyze_output_folder(
        output_dir,
        weber_minnhagen_C=None,
        bkt_n_bootstrap=args.bkt_bootstrap,
        bkt_rng_seed=args.bkt_rng_seed,
    )

    effective_wm_C, wm_fit = auto_select_weber_minnhagen_C(
        analyzed_by_L,
        provided_C=args.weber_minnhagen_C,
        fit_enabled=not bool(args.no_fit_weber_minnhagen_C),
        scan_lo=args.wm_scan_lo,
        scan_hi=args.wm_scan_hi,
        scan_points=args.wm_scan_points,
        min_sizes=args.wm_min_sizes,
        require_all_sizes=not bool(args.wm_allow_missing_sizes),
    )
    attach_weber_minnhagen_fit(analyzed_by_L, wm_fit)
    update_helicity_intersections_with_C(
        analyzed_by_L,
        weber_minnhagen_C=effective_wm_C,
        bkt_n_bootstrap=args.bkt_bootstrap,
        bkt_rng_seed=args.bkt_rng_seed,
    )

    analysis_report = build_analysis_report(analyzed_by_L)
    written: list[Path] = []
    written.extend(
        plot_standard_temperature_observables(
            analyzed_by_L,
            plots_dir,
        )
    )
    written.extend(
        plot_helicity_bkt_reference(
            analyzed_by_L,
            plots_dir,
            weber_minnhagen_C=effective_wm_C,
        )
    )
    written.extend(
        plot_helicity_diagnostics(
            analyzed_by_L,
            plots_dir,
        )
    )
    written.extend(
        plot_extra_observables(
            analyzed_by_L,
            plots_dir,
        )
    )
    summary_path = write_plot_summary(
        output_dir=output_dir,
        plots_dir=plots_dir,
        analyzed_by_L=analyzed_by_L,
        written=written,
        analysis_report=analysis_report,
    )
    print_analysis_report(
        output_dir=output_dir,
        plots_dir=plots_dir,
        analyzed_by_L=analyzed_by_L,
        written=written,
        summary_path=summary_path,
        report=analysis_report,
    )
    if args.show:
        plt.show()

if __name__ == "__main__":
    main()
