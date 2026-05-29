from __future__ import annotations

import argparse
import math
from typing import Any

import numpy as np

from ptmc.common.output import save_experiment_outputs
from ptmc.common.temperature_ladder import (
    make_temperature_ladder,
    temperature_ladder_diagnostics,
)
from ptmc.cpu.models.ising import IsingModel as CPUIsingModel
from ptmc.cpu.models.xy import XYModel as CPUXYModel
from ptmc.cpu.runner import ParallelTemperingCPU


def build_model(args: argparse.Namespace):
    if args.model == "ising":
        return CPUIsingModel(
            J=args.J,
            h=args.h,
            ordered_start=args.ordered_start,
        )
    return CPUXYModel(
        J=args.J,
        theta_step=args.theta_step,
        ordered_start=args.ordered_start,
    )


def run_cpu_experiment(args: argparse.Namespace) -> dict[str, Any]:
    model = build_model(args)
    temps = make_temperature_ladder(
        T_min=args.T_min,
        T_max=args.T_max,
        n_T=args.n_T,
        method=args.ladder_method,
        dense_near_tc=args.dense_near_tc,
        Tc=args.T_focus,
        tc_window=args.tc_window,
        tc_fraction=args.tc_fraction,
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

    results_by_L: dict[int, dict[str, Any]] = {}
    for index, L in enumerate(args.L):
        seed_for_this_L = int(args.seed) + index
        print(f"Running CPU PT benchmark for L={L} ...", flush=True)
        sim = ParallelTemperingCPU(
            L=L,
            temps=temps,
            n_equil_sweeps=args.n_equil_sweeps,
            n_measure_sweeps=args.n_measure_sweeps,
            model=model,
            sweeps_between_swaps=args.sweeps_between_swaps,
            record_stride=args.record_stride,
            seed=seed_for_this_L,
            energy_recompute_stride=args.energy_recompute_stride,
        )
        results_by_L[int(L)] = sim.run(
            record_during_equil=False,
            store_primary_histories=args.store_primary_histories,
            observable_n_blocks=args.observable_n_blocks,
        )

    return {
        "model_metadata": model.metadata(),
        "L_values": [int(L) for L in args.L],
        "temps": temps,
        "ladder_diagnostics": ladder_diagnostics,
        "parameters": {
            "backend": "cpu",
            "T_min": float(args.T_min),
            "T_max": float(args.T_max),
            "n_T": int(args.n_T),
            "ladder_method": str(args.ladder_method),
            "dense_near_tc": bool(args.dense_near_tc),
            "T_focus": float(args.T_focus),
            "tc_window": float(args.tc_window),
            "tc_fraction": float(args.tc_fraction),
            "n_equil_sweeps": int(args.n_equil_sweeps),
            "n_measure_sweeps": int(args.n_measure_sweeps),
            "sweeps_between_swaps": int(args.sweeps_between_swaps),
            "record_stride": int(args.record_stride),
            "rng_seed": int(args.seed),
            "energy_recompute_stride": int(args.energy_recompute_stride),
            "store_primary_histories": bool(args.store_primary_histories),
            "observable_n_blocks": int(args.observable_n_blocks),
        },
        "results_by_L": results_by_L,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small CPU parallel-tempering benchmark."
    )
    parser.add_argument("--model", choices=("ising", "xy"), default="ising")
    parser.add_argument("--L", type=int, nargs="+", default=[16])
    parser.add_argument("--T-min", dest="T_min", type=float, default=1.5)
    parser.add_argument("--T-max", dest="T_max", type=float, default=3.5)
    parser.add_argument("--n-T", dest="n_T", type=int, default=16)
    parser.add_argument("--ladder-method", default="beta")
    parser.add_argument("--dense-near-tc", action="store_true")
    parser.add_argument("--T-focus", dest="T_focus", type=float, default=2.269)
    parser.add_argument("--tc-window", type=float, default=0.4)
    parser.add_argument("--tc-fraction", type=float, default=0.5)
    parser.add_argument("--n-equil-sweeps", type=int, default=1_000)
    parser.add_argument("--n-measure-sweeps", type=int, default=2_000)
    parser.add_argument("--sweeps-between-swaps", type=int, default=5)
    parser.add_argument("--record-stride", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--h", type=float, default=0.0)
    parser.add_argument("--theta-step", type=float, default=math.pi / 2.0)
    parser.add_argument("--ordered-start", action="store_true")
    parser.add_argument("--energy-recompute-stride", type=int, default=100)
    parser.add_argument("--observable-n-blocks", type=int, default=20)
    parser.add_argument("--store-primary-histories", action="store_true")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment = run_cpu_experiment(args)
    model = build_model(args)
    output_dir = args.output_dir or f"outputs/{args.model}_cpu_benchmark"
    save_experiment_outputs(
        experiment,
        model=model,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
