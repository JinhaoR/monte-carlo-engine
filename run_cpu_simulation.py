from __future__ import annotations

from ptmc.cpu.experiment import run_pt_experiment as run_cpu_pt_experiment
from ptmc.cpu.models.ising import IsingModel


def main() -> None:
    model = IsingModel(
        J=1.0,
        h=0.0,
        ordered_start=False,
        output_prefix="ising2d_cpu",
    )

    run_cpu_pt_experiment(
        model=model,
        L_values=[16, 32],
        T_min=1.5,
        T_max=3.5,
        n_T=64,
        ladder_method="beta",
        dense_near_tc=True,
        T_focus=2.269185314213022,
        tc_window=0.4,
        tc_fraction=0.60,
        n_equil_sweeps=50_000,
        n_measure_sweeps=100_000,
        sweeps_between_swaps=10,
        record_stride=10,
        derived_observable_stride=50,
        rng_seed=1234,
        energy_recompute_stride=100,
        energy_drift_tolerance_per_site=1.0e-10,
        store_primary_histories=False,
        observable_n_blocks=20,
        output_dir="outputs/ising_cpu",
    )


if __name__ == "__main__":
    main()
