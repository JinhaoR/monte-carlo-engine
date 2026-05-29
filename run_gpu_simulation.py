from __future__ import annotations
import math
from ptmc.common.output import save_experiment_outputs
from ptmc.gpu.experiment import run_pt_experiment
from ptmc.gpu.models.xy import XYModel

def main() -> None:
    model = XYModel(
        J=1.0,
        ordered_start=False,
        output_prefix="xy2d",
    )
    experiment = run_pt_experiment(
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
        field_step=math.pi / 2.0,
        threads_per_block=128,
        energy_recompute_stride=0,
        store_primary_histories=False,
        observable_n_blocks=20,
    )
    save_experiment_outputs(
        experiment,
        model=model,
        output_dir="outputs/xy_bkt",
    )

if __name__ == "__main__":
    main()
