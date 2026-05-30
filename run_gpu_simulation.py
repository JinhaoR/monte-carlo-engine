from __future__ import annotations

from ptmc.gpu.experiment import run_pt_experiment as run_gpu_pt_experiment
from ptmc.gpu.models import TBGModel
from ptmc.gpu.models import SpinFrozenModel


def main() -> None:
    model = TBGModel(
    K=5.0,
    phase_step=1.0,
    amplitude_step=0.10,
    ordered_start=False,
    output_prefix="tbg2d_K5_gpu",
)

    run_gpu_pt_experiment(
        model=model,
        L_values=[16, 32, 64],
        T_min=0.5,
        T_max=0.7,
        n_T=60,
        ladder_method="beta",
        dense_near_tc=False,
        T_focus=0.58,
        tc_window=0.08,
        tc_fraction=0.70,
        n_equil_sweeps=100_000,
        n_measure_sweeps=200_000,
        sweeps_between_swaps=10,
        record_stride=10,
        derived_observable_stride=1,
        tracked_walkers=[0, 10, 20, 30, 40, 50],
        rng_seed=1234,
        threads_per_block=128,
        energy_recompute_stride=100,
        energy_drift_tolerance_per_site=1.0e-7,
        store_primary_histories=False,
        observable_n_blocks=20,
        output_dir="outputs/tbg_K5_gpu",
    )

if __name__ == "__main__":
    main()
