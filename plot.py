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
from analysis.bkt import helicity_reference_label, helicity_reference_line
from analysis.pipeline import analyze_run

RUN_FILE_GLOB = "*_L*.npz"
KEEP_FIGURES_OPEN = False

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
    return out

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

def infer_autocorrelation_keys(data: dict[str, Any]) -> list[str]:
    """
    Choose full-history arrays for autocorrelation analysis.

    Autocorrelation only makes sense if full measurement histories were saved.
    Block means alone are not enough.
    """
    candidates = [
        "energies",
        "energy",
        "order_parameter",
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

def analyze_output_folder(output_dir: Path) -> dict[int, dict[str, Any]]:
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
        obs = analyze_run(
            data,
            L=L,
            temps=data.get("temps"),
            energy_per_site=energy_per_site,
            order_parameter_per_site=order_parameter_per_site,
            record_stride=record_stride,
            autocorrelation_keys=autocorr_keys,
            extra_observable_specs=extra_specs,
        )
        obs["_source_file"] = str(path)
        obs["_params"] = params
        obs["_extra_observable_names"] = sorted(extra_specs.keys())
        obs["_autocorrelation_keys"] = autocorr_keys
        analyzed_by_L[L] = obs
    return dict(sorted(analyzed_by_L.items()))

# ============================================================
# Plot helpers
# ============================================================

def safe_filename_token(name: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name).strip().lower())
    return token.strip("._-") or "plot"

def has_temperature_curve(obs: dict[str, Any], key: str) -> bool:
    if key not in obs or "temps" not in obs:
        return False
    temps = np.asarray(obs["temps"])
    values = np.asarray(obs[key])
    return values.ndim == 1 and temps.ndim == 1 and values.shape == temps.shape

def finish_figure(fig: Any, out_path: Path) -> None:
    """
    Save a figure and close it unless interactive display was requested.
    """
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
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
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for L in available:
        obs = analyzed_by_L[L]
        temps = np.asarray(obs["temps"], dtype=np.float64)
        values = np.asarray(obs[key], dtype=np.float64)
        errors = None
        if err_key is not None and err_key in obs:
            maybe_errors = np.asarray(obs[err_key], dtype=np.float64)
            if maybe_errors.shape == values.shape:
                errors = maybe_errors
        if errors is None:
            ax.plot(temps, values, "o-", label=fr"$L={L}$")
        else:
            ax.errorbar(
                temps,
                values,
                yerr=errors,
                fmt="o-",
                capsize=2,
                label=fr"$L={L}$",
            )
    ax.set_xlabel(r"Temperature $T$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
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
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for L in available:
        values = np.asarray(analyzed_by_L[L][key], dtype=np.float64)
        x = np.arange(values.size)

        ax.plot(x, values, "o-", label=fr"$L={L}$")
    ax.set_xlabel("Index")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    finish_figure(fig, out_path)
    return True

# ============================================================
# Standard observable plots
# ============================================================

STANDARD_TEMPERATURE_PLOTS = [
    ("e", "e_err", r"Energy per site $e$", "Energy per site", "energy_per_site"),
    ("C", "C_err", r"Specific heat $C$", "Specific heat", "specific_heat"),
    ("m_abs", "m_abs_err", r"Z$_2$ order parameter $\langle |m| \rangle$", "Z2 order parameter", "z2_order_parameter"),
    ("chi", "chi_err", r"Susceptibility $\chi$", "Susceptibility", "susceptibility"),
    ("U4", "U4_err", r"Binder cumulant $U_4$", "Binder cumulant", "binder_cumulant"),
    ("Y", "Y_err", r"Helicity modulus $Y$", "Helicity modulus", "helicity_modulus"),
    ("helicity_Yx", "helicity_Yx_err", r"$Y_x$", "Helicity component Yx", "helicity_Yx"),
    ("helicity_Yy", "helicity_Yy_err", r"$Y_y$", "Helicity component Yy", "helicity_Yy"),
    ("helicity_Kx", "helicity_Kx_err", r"$K_x$", "Helicity Kx", "helicity_Kx"),
    ("helicity_Ky", "helicity_Ky_err", r"$K_y$", "Helicity Ky", "helicity_Ky"),
    ("helicity_Ix", "helicity_Ix_err", r"$I_x$", "Helicity Ix", "helicity_Ix"),
    ("helicity_Iy", "helicity_Iy_err", r"$I_y$", "Helicity Iy", "helicity_Iy"),
    ("helicity_Ix2", "helicity_Ix2_err", r"$I_x^2$", "Helicity Ix2", "helicity_Ix2"),
    ("helicity_Iy2", "helicity_Iy2_err", r"$I_y^2$", "Helicity Iy2", "helicity_Iy2"),
    (
        "helicity_Ix_mean_over_rms",
        None,
        r"$|\langle I_x\rangle| / \sqrt{\langle I_x^2\rangle}$",
        "Ix mean over RMS",
        "helicity_Ix_mean_over_rms",
    ),
    (
        "helicity_Iy_mean_over_rms",
        None,
        r"$|\langle I_y\rangle| / \sqrt{\langle I_y^2\rangle}$",
        "Iy mean over RMS",
        "helicity_Iy_mean_over_rms",
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
) -> list[Path]:
    """
    Plot helicity modulus with the BKT reference line.
    """
    available = [
        L for L, obs in analyzed_by_L.items()
        if has_temperature_curve(obs, "Y")
    ]
    if not available:
        return []
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    line_label_used = False
    for L in available:
        obs = analyzed_by_L[L]
        temps = np.asarray(obs["temps"], dtype=np.float64)
        Y = np.asarray(obs["Y"], dtype=np.float64)
        Y_err = np.asarray(obs.get("Y_err", np.full_like(Y, np.nan)))
        bkt = obs.get("bkt_intersection", {})
        wm_C = bkt.get("weber_minnhagen_C")
        if wm_C is not None and not np.isfinite(wm_C):
            wm_C = None
        reference = helicity_reference_line(
            temps,
            L=L,
            weber_minnhagen_C=wm_C,
        )
        ax.errorbar(
            temps,
            Y,
            yerr=Y_err if Y_err.shape == Y.shape else None,
            fmt="o-",
            capsize=2,
            label=fr"$L={L}$",
        )
        ax.plot(
            temps,
            reference,
            "--",
            color="0.35",
            alpha=0.7,
            label=helicity_reference_label(wm_C) if not line_label_used else None,
        )
        line_label_used = True
        if bkt.get("available", False):
            ax.axvline(
                float(bkt["T"]),
                color="0.35",
                linestyle=":",
                linewidth=1,
                alpha=0.6,
            )
    ax.set_xlabel(r"Temperature $T$")
    ax.set_ylabel(r"Helicity modulus $Y$")
    ax.set_title("Helicity modulus and BKT reference")
    ax.grid(alpha=0.3)
    ax.legend()
    out_path = plots_dir / "helicity_bkt_reference.png"
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
# Diagnostic plots
# ============================================================

def plot_swap_rate(
    analyzed_by_L: dict[int, dict[str, Any]],
    plots_dir: Path,
) -> list[Path]:
    """
    Plot PT swap acceptance rate versus temperature-edge midpoint.
    """
    available = [
        L for L, obs in analyzed_by_L.items()
        if "swap_rate" in obs and "temps" in obs
    ]
    if not available:
        return []
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for L in available:
        obs = analyzed_by_L[L]
        temps = np.asarray(obs["temps"], dtype=np.float64)
        rate = np.asarray(obs["swap_rate"], dtype=np.float64)
        if rate.ndim != 1 or rate.size == 0:
            continue
        if temps.ndim == 1 and temps.size == rate.size + 1:
            x = 0.5 * (temps[:-1] + temps[1:])
            xlabel = r"Temperature edge midpoint"
        else:
            x = np.arange(rate.size)
            xlabel = "Swap edge index"

        ax.plot(x, rate, "o-", label=fr"$L={L}$")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Swap acceptance rate")
    ax.set_title("Parallel-tempering swap acceptance")
    ax.grid(alpha=0.3)
    ax.legend()
    out_path = plots_dir / "swap_acceptance_rate.png"
    finish_figure(fig, out_path)
    return [out_path]

def plot_local_acceptance(
    analyzed_by_L: dict[int, dict[str, Any]],
    plots_dir: Path,
) -> list[Path]:
    """
    Plot local acceptance rates if available.

    These are plotted versus index because the axis meaning is model-dependent.
    """
    written: list[Path] = []
    for key in sorted(
        {
            key
            for obs in analyzed_by_L.values()
            for key in obs
            if key.endswith("_acceptance_rate") and key != "swap_rate"
        }
    ):
        out_path = plots_dir / f"{safe_filename_token(key)}.png"
        ok = plot_index_curve(
            analyzed_by_L,
            key=key,
            ylabel=key.replace("_", " "),
            title=key.replace("_", " ").title(),
            out_path=out_path,
        )
        if ok:
            written.append(out_path)
    return written

def plot_round_trip_counts(
    analyzed_by_L: dict[int, dict[str, Any]],
    plots_dir: Path,
) -> list[Path]:
    """
    Plot round-trip counts for PT walkers.
    """
    return _plot_round_trip_helper(analyzed_by_L, plots_dir)

def _plot_round_trip_helper(
    analyzed_by_L: dict[int, dict[str, Any]],
    plots_dir: Path,
) -> list[Path]:
    written: list[Path] = []
    for key, title in [
        ("round_trip_counts", "Round-trip counts"),
        ("commute_counts", "Single-edge commute counts"),
    ]:
        out_path = plots_dir / f"{key}.png"
        ok = plot_index_curve(
            analyzed_by_L,
            key=key,
            ylabel=title,
            title=title,
            out_path=out_path,
        )
        if ok:
            written.append(out_path)
    return written

def plot_round_trip_duration_histogram(
    analyzed_by_L: dict[int, dict[str, Any]],
    plots_dir: Path,
) -> list[Path]:
    """
    Plot histogram of round-trip durations.
    """
    available = [
        L for L, obs in analyzed_by_L.items()
        if "round_trip_durations" in obs
        and np.asarray(obs["round_trip_durations"]).size > 0
    ]
    if not available:
        return []
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for L in available:
        durations = np.asarray(
            analyzed_by_L[L]["round_trip_durations"],
            dtype=np.float64,
        )
        durations = durations[np.isfinite(durations)]
        if durations.size == 0:
            continue
        ax.hist(
            durations,
            bins="auto",
            histtype="step",
            label=fr"$L={L}$",
        )
    ax.set_xlabel("Round-trip duration")
    ax.set_ylabel("Count")
    ax.set_title("PT round-trip duration histogram")
    ax.grid(alpha=0.3)
    ax.legend()
    out_path = plots_dir / "round_trip_durations.png"
    finish_figure(fig, out_path)
    return [out_path]

def plot_autocorrelation_times(
    analyzed_by_L: dict[int, dict[str, Any]],
    plots_dir: Path,
) -> list[Path]:
    """
    Plot tau_int curves produced by diagnostics.
    """
    tau_keys = sorted(
        {
            key
            for obs in analyzed_by_L.values()
            for key in obs
            if key.endswith("_tau_int")
        }
    )
    written: list[Path] = []
    for key in tau_keys:
        out_path = plots_dir / f"{safe_filename_token(key)}.png"
        ok = plot_temperature_curve(
            analyzed_by_L,
            key=key,
            err_key=None,
            ylabel=r"Integrated autocorrelation time $\tau_{\mathrm{int}}$",
            title=key.replace("_", " ").title(),
            out_path=out_path,
        )
        if ok:
            written.append(out_path)
    return written

def plot_energy_drift(
    analyzed_by_L: dict[int, dict[str, Any]],
    plots_dir: Path,
) -> list[Path]:
    """
    Plot energy drift diagnostics if available.
    """
    written: list[Path] = []
    for key in [
        "energy_drift_abs_max_by_walker",
        "energy_drift_recompute_count",
    ]:
        out_path = plots_dir / f"{safe_filename_token(key)}.png"
        ok = plot_index_curve(
            analyzed_by_L,
            key=key,
            ylabel=key.replace("_", " "),
            title=key.replace("_", " ").title(),
            out_path=out_path,
        )
        if ok:
            written.append(out_path)
    return written

def plot_diagnostics(
    analyzed_by_L: dict[int, dict[str, Any]],
    plots_dir: Path,
) -> list[Path]:
    """
    Plot all available simulation diagnostics.
    """
    written: list[Path] = []
    written.extend(plot_swap_rate(analyzed_by_L, plots_dir))
    written.extend(plot_local_acceptance(analyzed_by_L, plots_dir))
    written.extend(plot_round_trip_counts(analyzed_by_L, plots_dir))
    written.extend(plot_round_trip_duration_histogram(analyzed_by_L, plots_dir))
    written.extend(plot_autocorrelation_times(analyzed_by_L, plots_dir))
    written.extend(plot_energy_drift(analyzed_by_L, plots_dir))
    return written

# ============================================================
# Summary
# ============================================================

def write_plot_summary(
    *,
    output_dir: Path,
    plots_dir: Path,
    analyzed_by_L: dict[int, dict[str, Any]],
    written: list[Path],
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
        "runs": {
            str(L): {
                "source_file": obs.get("_source_file"),
                "extra_observables": obs.get("_extra_observable_names", []),
                "autocorrelation_keys": obs.get("_autocorrelation_keys", []),
                "bkt_intersection": obs.get("bkt_intersection"),
            }
            for L, obs in analyzed_by_L.items()
        },
    }
    out_path = plots_dir / "plot_summary.json"
    out_path.write_text(
        json.dumps(summary, indent=2),
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
        "--show",
        action="store_true",
        help="Show plots interactively after saving.",
    )
    return parser.parse_args()


def main() -> None:
    global KEEP_FIGURES_OPEN
    args = parse_args()
    KEEP_FIGURES_OPEN = bool(args.show)
    output_dir = resolve_output_dir(args.output_folder)
    if args.plots_dir is None:
        plots_dir = output_dir / "plots"
    else:
        plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    analyzed_by_L = analyze_output_folder(output_dir)
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
        )
    )
    written.extend(
        plot_extra_observables(
            analyzed_by_L,
            plots_dir,
        )
    )
    written.extend(
        plot_diagnostics(
            analyzed_by_L,
            plots_dir,
        )
    )
    summary_path = write_plot_summary(
        output_dir=output_dir,
        plots_dir=plots_dir,
        analyzed_by_L=analyzed_by_L,
        written=written,
    )
    print(f"Plotted output folder: {output_dir}")
    print(f"Wrote {len(written)} plot files to: {plots_dir}")
    print(f"Wrote summary: {summary_path}")
    if args.show:
        plt.show()

if __name__ == "__main__":
    main()
