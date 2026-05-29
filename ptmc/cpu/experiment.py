from __future__ import annotations
from typing import Any, Iterable
import numpy as np

from ptmc.common.temperature_ladder import (
    make_temperature_ladder,
    temperature_ladder_diagnostics,
)
from ptmc.cpu.interface import BaseCPUModel, validate_cpu_model
from ptmc.cpu.runner import ParallelTemperingCPU


def run_pt_experiment(
    *,
    model: BaseCPUModel,
    L_values: Iterable[int],
    T_min: float,
    T_max: float,
    n_T: int,
    ladder_method: str = "beta",
    dense_near_tc: bool = False,
    T_focus: float = 1.0,
    tc_window: float = 0.05,
    tc_fraction: float = 0.50,
    n_equil_sweeps: int = 5_000,
    n_measure_sweeps: int = 10_000,
    sweeps_between_swaps: int = 1,
    record_stride: int = 10,
    derived_observable_stride: int = 1,
    rng_seed: int = 1234,
    energy_recompute_stride: int = 0,
    energy_drift_tolerance_per_site: float | None = 1.0e-10,
    store_primary_histories: bool = True,
    observable_n_blocks: int = 20,
) -> dict[str, Any]:
    """Run one CPU parallel-tempering experiment for several system sizes."""

    model = validate_cpu_model(model)

    L_values = [int(L) for L in L_values]
    if not L_values:
        raise ValueError("L_values must contain at least one system size.")

    for L in L_values:
        if L <= 0:
            raise ValueError(f"L={L} is not positive.")
        model.validate_lattice(L)

    temps = make_temperature_ladder(
        T_min=T_min,
        T_max=T_max,
        n_T=n_T,
        method=ladder_method,
        dense_near_tc=dense_near_tc,
        Tc=T_focus,
        tc_window=tc_window,
        tc_fraction=tc_fraction,
    )

    ladder_diagnostics = temperature_ladder_diagnostics(temps)

    print(
        "Temperature ladder diagnostics: "
        f"min dT={ladder_diagnostics['min_delta_T']:.6f}, "
        f"max dT={ladder_diagnostics['max_delta_T']:.6f}, "
        f"gap ratio={ladder_diagnostics['delta_T_gap_ratio']:.3f}, "
        f"max adjacent ratio={ladder_diagnostics['max_adjacent_delta_ratio']:.3f}",
        flush=True,
    )

    if ladder_diagnostics["max_adjacent_delta_ratio"] > 1.5:
        print(
            "Warning: adjacent temperature spacings change abruptly. "
            "This may reduce swap efficiency.",
            flush=True,
        )

    results_by_L: dict[int, dict[str, Any]] = {}

    for index, L in enumerate(L_values):
        seed_for_this_L = int(rng_seed) + index

        print(f"Running CPU PT simulation for L={L} ...", flush=True)

        sim = ParallelTemperingCPU(
            L=L,
            temps=temps,
            n_equil_sweeps=n_equil_sweeps,
            n_measure_sweeps=n_measure_sweeps,
            model=model,
            sweeps_between_swaps=sweeps_between_swaps,
            record_stride=record_stride,
            seed=seed_for_this_L,
            energy_recompute_stride=energy_recompute_stride,
            energy_drift_tolerance_per_site=energy_drift_tolerance_per_site,
        )

        result = sim.run(
            record_during_equil=False,
            derived_observable_stride=derived_observable_stride,
            store_primary_histories=store_primary_histories,
            observable_n_blocks=observable_n_blocks,
        )

        results_by_L[L] = result

    return {
        "model_metadata": model.metadata(),
        "L_values": L_values,
        "temps": temps,
        "ladder_diagnostics": ladder_diagnostics,
        "parameters": {
            "backend": "cpu",
            "T_min": float(T_min),
            "T_max": float(T_max),
            "n_T": int(n_T),
            "ladder_method": str(ladder_method),
            "dense_near_tc": bool(dense_near_tc),
            "T_focus": float(T_focus),
            "tc_window": float(tc_window),
            "tc_fraction": float(tc_fraction),
            "n_equil_sweeps": int(n_equil_sweeps),
            "n_measure_sweeps": int(n_measure_sweeps),
            "sweeps_between_swaps": int(sweeps_between_swaps),
            "record_stride": int(record_stride),
            "derived_observable_stride": int(derived_observable_stride),
            "rng_seed": int(rng_seed),
            "energy_recompute_stride": int(energy_recompute_stride),
            "energy_drift_tolerance_per_site": (
                None
                if energy_drift_tolerance_per_site is None
                else float(energy_drift_tolerance_per_site)
            ),
            "store_primary_histories": bool(store_primary_histories),
            "observable_n_blocks": int(observable_n_blocks),
        },
        "results_by_L": results_by_L,
    }