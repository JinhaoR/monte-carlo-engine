from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
from numba import cuda

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ptmc.common.temperature_ladder import make_temperature_ladder
from ptmc.gpu.models import SpinFrozenModel
from ptmc.gpu.runner import ParallelTemperingGPU


def positive_int(value: str) -> int:
    out = int(value)
    if out <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return out


def nonnegative_int(value: str) -> int:
    out = int(value)
    if out < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Focused CUDA profiling target for SpinFrozenModel.",
    )
    parser.add_argument("--L", type=positive_int, default=64)
    parser.add_argument("--n-temps", type=positive_int, default=32)
    parser.add_argument("--T-min", type=float, default=0.4)
    parser.add_argument("--T-max", type=float, default=1.2)
    parser.add_argument("--K", type=float, default=1.0)
    parser.add_argument("--field-step", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--threads-per-block", type=positive_int, default=128)
    parser.add_argument("--warmup-sweeps", type=nonnegative_int, default=20)
    parser.add_argument("--profile-sweeps", type=positive_int, default=200)
    parser.add_argument("--swaps-between", type=positive_int, default=10)
    parser.add_argument(
        "--mode",
        choices=("sweeps", "sweeps-and-observables"),
        default="sweeps",
        help="Profile only local/PT updates, or include compact observable records.",
    )
    parser.add_argument("--derived-observable-stride", type=positive_int, default=10)
    parser.add_argument("--observable-n-blocks", type=positive_int, default=20)
    parser.add_argument(
        "--ordered-start",
        action="store_true",
        help="Start from psi_+=1, psi_-=0 instead of random local chirality.",
    )
    parser.add_argument(
        "--cuda-profile-api",
        action="store_true",
        help="Bracket the measured region with cudaProfilerStart/Stop.",
    )
    return parser


def maybe_attempt_swap(sim: ParallelTemperingGPU, sweep_index: int, stride: int) -> None:
    if stride > 0 and (sweep_index + 1) % stride == 0:
        sim._attempt_swaps()


def run_warmup(sim: ParallelTemperingGPU, n_sweeps: int, swaps_between: int) -> None:
    for sweep in range(n_sweeps):
        sim._advance_model_state()
        maybe_attempt_swap(sim, sweep, swaps_between)


def allocate_profile_observables(
    sim: ParallelTemperingGPU,
    *,
    n_sweeps: int,
    derived_stride: int,
    observable_n_blocks: int,
) -> None:
    n_derived = (n_sweeps + derived_stride - 1) // derived_stride
    sim._allocate_measurement_storage(
        n_sweeps,
        n_derived,
        False,
        observable_n_blocks,
    )


def warmup_observable_kernels(sim: ParallelTemperingGPU) -> None:
    """
    Compile/lazy-load observable kernels before the profiled region.

    The profiling target is not used for statistical output, so writing into
    column zero during warmup is harmless and keeps timings focused on runtime.
    """
    sim._record_primary_observables_to_output(0)
    sim._record_derived_observables_to_output(0)


def run_profiled_region(
    sim: ParallelTemperingGPU,
    *,
    n_sweeps: int,
    swaps_between: int,
    mode: str,
    derived_stride: int,
) -> None:
    derived_col = 0
    for sweep in range(n_sweeps):
        sim._advance_model_state()
        maybe_attempt_swap(sim, sweep, swaps_between)
        if mode == "sweeps-and-observables":
            sim._record_primary_observables_to_output(sweep)
            if sweep % derived_stride == 0:
                sim._record_derived_observables_to_output(derived_col)
                derived_col += 1


def main() -> None:
    args = build_parser().parse_args()
    if not cuda.is_available():
        raise RuntimeError("CUDA is not available; Nsight profiling needs a GPU run.")

    temps = make_temperature_ladder(
        T_min=args.T_min,
        T_max=args.T_max,
        n_T=args.n_temps,
        method="beta",
    )
    model = SpinFrozenModel(
        K=args.K,
        field_step=args.field_step,
        ordered_start=args.ordered_start,
        output_prefix="spin_frozen_profile",
    )
    rng = np.random.default_rng(args.seed)
    sim = ParallelTemperingGPU(
        L=args.L,
        temps=temps,
        n_equil_sweeps=max(1, args.warmup_sweeps),
        n_measure_sweeps=args.profile_sweeps,
        model=model,
        sweeps_between_swaps=args.swaps_between,
        record_stride=10,
        seed=args.seed,
        rng=rng,
        threads_per_block=args.threads_per_block,
        energy_recompute_stride=0,
    )

    if args.mode == "sweeps-and-observables":
        allocate_profile_observables(
            sim,
            n_sweeps=args.profile_sweeps,
            derived_stride=args.derived_observable_stride,
            observable_n_blocks=args.observable_n_blocks,
        )

    run_warmup(sim, args.warmup_sweeps, args.swaps_between)
    if args.mode == "sweeps-and-observables":
        warmup_observable_kernels(sim)
    reset_stats = getattr(sim.runtime, "reset_local_acceptance_stats", None)
    if reset_stats is not None:
        reset_stats()
    cuda.synchronize()

    if args.cuda_profile_api:
        cuda.profile_start()
    t0 = time.perf_counter()
    run_profiled_region(
        sim,
        n_sweeps=args.profile_sweeps,
        swaps_between=args.swaps_between,
        mode=args.mode,
        derived_stride=args.derived_observable_stride,
    )
    cuda.synchronize()
    elapsed = time.perf_counter() - t0
    if args.cuda_profile_api:
        cuda.profile_stop()
        cuda.synchronize()

    attempts = sim.runtime.d_local_update_attempts.copy_to_host()
    accepted = sim.runtime.d_local_update_acceptance.copy_to_host()
    attempted_total = int(np.sum(attempts))
    accepted_total = int(np.sum(accepted))
    acceptance = accepted_total / attempted_total if attempted_total else float("nan")
    sweeps_per_second = args.profile_sweeps / elapsed if elapsed > 0.0 else float("nan")

    print(
        "SpinFrozen profile "
        f"L={args.L} R={args.n_temps} K={args.K:g} "
        f"field_step={args.field_step:g} mode={args.mode}",
        flush=True,
    )
    print(
        f"elapsed_s={elapsed:.6f} "
        f"sweeps_per_second={sweeps_per_second:.6f} "
        f"local_acceptance={acceptance:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
