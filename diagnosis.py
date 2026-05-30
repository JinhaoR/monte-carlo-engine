#!/usr/bin/env python3
"""
Plot PTMC run diagnostics.

Usage
-----
From the project root:
    python diagnosis.py outputs/spin_frozen_K4_gpu

The script writes diagnostic figures to:
    outputs/spin_frozen_K4_gpu/diagnosis/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import plot as plot_core
from analysis.autocorrelation_plots import plot_autocorrelation_diagnostics


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
    colors = plot_core._multi_L_colors(available)
    fig, ax = plt.subplots(figsize=plot_core.FIGSIZE)
    for L in available:
        obs = analyzed_by_L[L]
        temps = np.asarray(obs["temps"], dtype=np.float64)
        rate = np.asarray(obs["swap_rate"], dtype=np.float64)
        if temps.size == rate.size + 1:
            x = 0.5 * (temps[:-1] + temps[1:])
            xlabel = r"Temperature-edge midpoint $T$"
        else:
            x = np.arange(rate.size)
            xlabel = "Swap edge index"
        plot_core._multi_L_plot(
            ax,
            x,
            rate,
            L=L,
            color=colors.get(int(L)),
        )
    plot_core._format_axes(
        ax,
        xlabel=xlabel,
        ylabel="Swap acceptance rate",
        title="Parallel-tempering swap acceptance",
    )
    plot_core._legend(ax)
    out_path = plots_dir / "swap_acceptance_rate.png"
    plot_core.finish_figure(fig, out_path)
    return [out_path]


def plot_local_acceptance(
    analyzed_by_L: dict[int, dict[str, Any]],
    plots_dir: Path,
) -> list[Path]:
    """
    Plot model-update acceptance rates.
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
        out_path = plots_dir / f"{plot_core.safe_filename_token(key)}.png"
        ok = plot_core.plot_index_curve(
            analyzed_by_L,
            key=key,
            ylabel=key.replace("_", " "),
            title=key.replace("_", " ").title(),
            out_path=out_path,
        )
        if ok:
            written.append(out_path)
    return written


def plot_walker_positions(
    analyzed_by_L: dict[int, dict[str, Any]],
    plots_dir: Path,
) -> list[Path]:
    """
    Plot representative PT walker trajectories through the temperature ladder.
    """
    available = [
        L for L, obs in analyzed_by_L.items()
        if "label_positions" in obs
        and np.asarray(obs["label_positions"]).ndim == 2
        and np.asarray(obs["label_positions"]).size > 0
    ]
    if not available:
        return []

    L = max(int(value) for value in available)
    obs = analyzed_by_L[L]
    positions = np.asarray(obs["label_positions"], dtype=np.int64)
    n_records, n_walkers = positions.shape
    record_stride = int(obs.get("label_position_record_stride", 1))
    x = (np.arange(n_records, dtype=np.float64) + 1.0) * record_stride
    n_traces = max(1, int(np.ceil(0.25 * n_walkers)))
    walker_indices = np.arange(0, n_walkers, 4, dtype=np.int64)
    if walker_indices.size > n_traces:
        walker_indices = walker_indices[:n_traces]

    fig, ax = plt.subplots(figsize=(7.2, 4.9))
    cmap = plt.get_cmap("rainbow")
    color_positions = np.linspace(0.02, 0.98, max(len(walker_indices), 2))
    for i, walker in enumerate(walker_indices):
        ax.plot(
            x,
            positions[:, walker],
            color=cmap(color_positions[i]),
            linewidth=0.75,
            alpha=0.72,
        )
    ax.axhline(0, color="0.25", linestyle=":", linewidth=0.9, alpha=0.7)
    ax.axhline(
        n_walkers - 1,
        color="0.25",
        linestyle=":",
        linewidth=0.9,
        alpha=0.7,
    )
    ax.set_ylim(-0.5, max(n_walkers - 0.5, 0.5))
    plot_core._format_axes(
        ax,
        xlabel="Swap attempt index",
        ylabel="Temperature slot",
        title=fr"$L={L}$ walker slot trajectories",
        integer_x=True,
    )
    out_path = plots_dir / "walker_positions.png"
    plot_core.finish_figure(fig, out_path)
    return [out_path]


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
    colors = plot_core._multi_L_colors(available)
    fig, ax = plt.subplots(figsize=plot_core.FIGSIZE)
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
            color=colors.get(int(L)),
            linewidth=plot_core.MULTI_L_LINEWIDTH,
            alpha=0.95,
            label=fr"$L={L}$",
        )
    plot_core._format_axes(
        ax,
        xlabel="Round-trip duration",
        ylabel="Count",
        title="PT round-trip duration histogram",
    )
    plot_core._legend(ax)
    out_path = plots_dir / "round_trip_durations.png"
    plot_core.finish_figure(fig, out_path)
    return [out_path]


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
        out_path = plots_dir / f"{plot_core.safe_filename_token(key)}.png"
        ok = plot_core.plot_index_curve(
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
    written.extend(plot_walker_positions(analyzed_by_L, plots_dir))
    written.extend(plot_round_trip_duration_histogram(analyzed_by_L, plots_dir))
    written.extend(plot_autocorrelation_diagnostics(analyzed_by_L, plots_dir))
    written.extend(plot_energy_drift(analyzed_by_L, plots_dir))
    return written


def write_diagnosis_summary(
    *,
    output_dir: Path,
    plots_dir: Path,
    analyzed_by_L: dict[int, dict[str, Any]],
    written: list[Path],
    analysis_report: dict[str, Any],
) -> Path:
    """
    Write a compact JSON summary of diagnostic plots and inputs.
    """
    summary = {
        "output_dir": str(output_dir),
        "diagnosis_dir": str(plots_dir),
        "L_values": sorted(int(L) for L in analyzed_by_L),
        "n_plots": len(written),
        "plots": [path.name for path in written],
        "analysis": analysis_report,
        "runs": {
            str(L): {
                "source_file": obs.get("_source_file"),
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
            }
            for L, obs in analyzed_by_L.items()
        },
    }
    out_path = plots_dir / "diagnosis_summary.json"
    out_path.write_text(
        json.dumps(plot_core._json_clean(summary), indent=2),
        encoding="utf-8",
    )
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot PTMC diagnostic output folder.",
    )
    parser.add_argument(
        "output_folder",
        help=(
            "Output folder to diagnose. Accepts either 'outputs/name' "
            "or just 'name'."
        ),
    )
    parser.add_argument(
        "--plots-dir",
        default=None,
        help=(
            "Directory for diagnostic plots. Defaults to "
            "'<output_folder>/diagnosis'."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Keep matplotlib figures open after writing files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_core.KEEP_FIGURES_OPEN = bool(args.show)
    output_dir = plot_core.resolve_output_dir(args.output_folder)
    if args.plots_dir is None:
        plots_dir = output_dir / "diagnosis"
    else:
        plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    analyzed_by_L = plot_core.analyze_output_folder(output_dir)
    effective_wm_C, wm_fit = plot_core.auto_select_weber_minnhagen_C(
        analyzed_by_L,
    )
    plot_core.attach_weber_minnhagen_fit(analyzed_by_L, wm_fit)
    plot_core.update_helicity_intersections_with_C(
        analyzed_by_L,
        weber_minnhagen_C=effective_wm_C,
    )
    analysis_report = plot_core.build_analysis_report(analyzed_by_L)
    written = plot_diagnostics(analyzed_by_L, plots_dir)
    summary_path = write_diagnosis_summary(
        output_dir=output_dir,
        plots_dir=plots_dir,
        analyzed_by_L=analyzed_by_L,
        written=written,
        analysis_report=analysis_report,
    )
    plot_core.print_analysis_report(
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
