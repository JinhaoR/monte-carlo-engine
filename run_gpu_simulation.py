from __future__ import annotations
from ptmc.gpu.experiment import run_pt_experiment as run_gpu_pt_experiment
from ptmc.gpu.models.xy import XYModel
from ptmc.gpu.models.ising import IsingModel

def main() -> None:
    model = IsingModel(
        J=1.0,
        h=0.0,
        ordered_start=False,
        output_prefix="ising2d_gpu",
    )
    run_gpu_pt_experiment(
        model=model,
        L_values=[16, 32, 64],
        T_min=1.5,
        T_max=3,
        n_T=100,
        ladder_method="beta",
        dense_near_tc=True,
        T_focus=2.269185314213022,
        tc_window=0.4,
        tc_fraction=0.60,
        n_equil_sweeps=200_000,
        n_measure_sweeps=200_000,
        sweeps_between_swaps=10,
        record_stride=10,
        derived_observable_stride=5,
        rng_seed=1234,
        energy_recompute_stride=100,
        energy_drift_tolerance_per_site=1.0e-10,
        store_primary_histories=True,
        observable_n_blocks=20,
        output_dir="outputs/ising_gpu",
    )


    """model = XYModel(
        J=1.0,
        theta_step=1.5707963267948966,
        ordered_start=False,
        output_prefix="xy2d",
    )
    experiment = run_gpu_pt_experiment(
        model=model,
        L_values=[16, 32, 64],
        T_min=0.6,
        T_max=1.2,
        n_T=64,
        ladder_method="beta",
        dense_near_tc=True,
        T_focus=0.893,
        tc_window=0.25,
        tc_fraction=0.60,
        n_equil_sweeps=50_000,
        n_measure_sweeps=100_000,
        sweeps_between_swaps=10,
        record_stride=10,
        derived_observable_stride=50,
        rng_seed=1234,
        threads_per_block=128,
        energy_recompute_stride=0,
        store_primary_histories=False,
        observable_n_blocks=20,
    )"""

if __name__ == "__main__":
    main()
