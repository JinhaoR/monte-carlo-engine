from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ptmc.common.output import (
    finish_experiment_output,
    local_timestamp,
    save_l_output,
    start_experiment_output,
)
from ptmc.common.temperature_ladder import (
    make_temperature_ladder,
    temperature_ladder_diagnostics,
)
from ptmc.gpu.interface import BasePTModel, validate_pt_model
from ptmc.gpu.runner import ParallelTemperingGPU

def run_pt_experiment(
    *,
    model: BasePTModel,
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
    threads_per_block: int = 128,
    energy_recompute_stride: int = 0,
    energy_drift_tolerance_per_site: float | None = 1.0e-5,
    store_primary_histories: bool = True,
    observable_n_blocks: int = 20,
    output_dir: str | Path | None = None,
    output_prefix: str | None = None,
) -> dict[str, Any]:
    """
    Run one parallel tempering experiment for several system sizes.
    This function coordinates many single L simulations.
    """
    started_at = local_timestamp()
    model = validate_pt_model(model)
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
    model_metadata = model.metadata()
    parameters = {
        "backend": "gpu",
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
        "threads_per_block": int(threads_per_block),
        "energy_recompute_stride": int(energy_recompute_stride),
        "energy_drift_tolerance_per_site": (
            None
            if energy_drift_tolerance_per_site is None
            else float(energy_drift_tolerance_per_site)
        ),
        "store_primary_histories": bool(store_primary_histories),
        "observable_n_blocks": int(observable_n_blocks),
    }
    output_state = None
    if output_dir is not None:
        output_state = start_experiment_output(
            model=model,
            output_dir=output_dir,
            output_prefix=output_prefix,
            L_values=L_values,
            temps=temps,
            ladder_diagnostics=ladder_diagnostics,
            parameters=parameters,
            started_at=started_at,
        )
    results_by_L: dict[int, dict[str, Any]] = {}
    run_times: dict[str, dict[str, Any]] = {}
    for index, L in enumerate(L_values):
        seed_for_this_L = int(rng_seed) + index
        L_started_at = local_timestamp()
        print(
            f"Running PT simulation for L={L} "
            f"(started {L_started_at}) ...",
            flush=True,
        )
        rng = np.random.default_rng(seed_for_this_L)
        sim = ParallelTemperingGPU(
            L=L,
            temps=temps,
            n_equil_sweeps=n_equil_sweeps,
            n_measure_sweeps=n_measure_sweeps,
            model=model,
            sweeps_between_swaps=sweeps_between_swaps,
            record_stride=record_stride,
            seed=seed_for_this_L,
            rng=rng,
            threads_per_block=threads_per_block,
            energy_recompute_stride=energy_recompute_stride,
            energy_drift_tolerance_per_site=energy_drift_tolerance_per_site,
        )
        result = sim.run(
            record_during_equil=False,
            derived_observable_stride=derived_observable_stride,
            store_primary_histories=store_primary_histories,
            observable_n_blocks=observable_n_blocks,
        )
        L_completed_at = local_timestamp()
        run_times[str(int(L))] = {
            "L": int(L),
            "started_at": L_started_at,
            "completed_at": L_completed_at,
        }
        results_by_L[L] = result
        if output_state is not None:
            out_dir, out_prefix, manifest_path, manifest = output_state
            save_l_output(
                output_dir=out_dir,
                output_prefix=out_prefix,
                manifest_path=manifest_path,
                manifest=manifest,
                model_metadata=model_metadata,
                parameters=parameters,
                L=L,
                result=result,
                started_at=L_started_at,
                completed_at=L_completed_at,
            )

    completed_at = local_timestamp()
    if output_state is not None:
        _, _, manifest_path, manifest = output_state
        finish_experiment_output(
            manifest_path=manifest_path,
            manifest=manifest,
            completed_at=completed_at,
        )
    return {
        "started_at": started_at,
        "completed_at": completed_at,
        "model_metadata": model_metadata,
        "L_values": L_values,
        "temps": temps,
        "ladder_diagnostics": ladder_diagnostics,
        "parameters": parameters,
        "runs": run_times,
        "results_by_L": results_by_L,
    }
