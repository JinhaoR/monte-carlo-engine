# Run Templates

Copy one block into `run_cpu_simulation.py` or `run_gpu_simulation.py`, then edit
the values for the experiment you want. The model block contains model-specific
parameters. The `run_*_pt_experiment(...)` block contains parallel-tempering,
measurement, diagnostics, and output settings.

## GPU Ising

```python
from ptmc.gpu.experiment import run_pt_experiment as run_gpu_pt_experiment
from ptmc.gpu.models.ising import IsingModel


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
    T_max=3.0,
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
    threads_per_block=128,
    energy_recompute_stride=100,
    energy_drift_tolerance_per_site=1.0e-10,
    store_primary_histories=True,
    observable_n_blocks=20,
    output_dir="outputs/ising_gpu",
)
```

## CPU Ising

```python
from ptmc.cpu.experiment import run_pt_experiment as run_cpu_pt_experiment
from ptmc.cpu.models.ising import IsingModel


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
```

## GPU XY

```python
import math

from ptmc.gpu.experiment import run_pt_experiment as run_gpu_pt_experiment
from ptmc.gpu.models.xy import XYModel


model = XYModel(
    J=1.0,
    theta_step=math.pi / 2.0,
    ordered_start=False,
    output_prefix="xy2d_gpu",
)

run_gpu_pt_experiment(
    model=model,
    L_values=[32, 64, 128],
    T_min=0.8,
    T_max=1,
    n_T=100,
    ladder_method="beta",
    dense_near_tc=False,
    T_focus=0.893,
    tc_window=0.25,
    tc_fraction=0.60,
    n_equil_sweeps=400_000,
    n_measure_sweeps=400_000,
    sweeps_between_swaps=10,
    record_stride=10,
    derived_observable_stride=10,
    rng_seed=1234,
    threads_per_block=128,
    energy_recompute_stride=0,
    store_primary_histories=True,
    observable_n_blocks=20,
    output_dir="outputs/xy_gpu",
)
```

## GPU TBG

```python
from ptmc.gpu.experiment import run_pt_experiment as run_gpu_pt_experiment
from ptmc.gpu.models import TBGModel


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
    n_T=100,
    ladder_method="beta",
    dense_near_tc=False,
    T_focus=0.58,
    tc_window=0.08,
    tc_fraction=0.70,
    n_equil_sweeps=300_000,
    n_measure_sweeps=240_000,
    sweeps_between_swaps=32,
    record_stride=10,
    derived_observable_stride=10,
    tracked_walkers=[0, 20]
    rng_seed=1234,
    threads_per_block=128,
    energy_recompute_stride=100,
    energy_drift_tolerance_per_site=1.0e-7,
    store_primary_histories=True,
    observable_n_blocks=20,
    output_dir="outputs/tbg_K5_gpu",
)
```

## GPU Spin Frozen

```python
from ptmc.gpu.experiment import run_pt_experiment as run_gpu_pt_experiment
from ptmc.gpu.models import SpinFrozenModel


model = SpinFrozenModel(
        K=4.0,
        field_step=0.15,
        ordered_start=False,
        output_prefix="spin_frozen2d_K4_gpu",
    )

    run_gpu_pt_experiment(
        model=model,
        L_values=[16, 32, 64],
        T_min=2.3,
        T_max=2.7,
        n_T=40,
        ladder_method="beta",
        dense_near_tc=False,
        T_focus=0.8,
        tc_window=0.2,
        tc_fraction=0.60,
        n_equil_sweeps=50_000,
        n_measure_sweeps=100_000,
        sweeps_between_swaps=10,
        record_stride=10,
        derived_observable_stride=1,
        tracked_walkers=[0, 20],
        rng_seed=1234,
        threads_per_block=128,
        energy_recompute_stride=1_000,
        energy_drift_tolerance_per_site=1.0e-6,
        store_primary_histories=True,
        observable_n_blocks=10,
        output_dir="outputs/spin_frozen_K4_gpu",
    )
```

## CPU XY

```python
import math

from ptmc.cpu.experiment import run_pt_experiment as run_cpu_pt_experiment
from ptmc.cpu.models.xy import XYModel


model = XYModel(
    J=1.0,
    theta_step=math.pi / 2.0,
    ordered_start=False,
    output_prefix="xy2d_cpu",
)

run_cpu_pt_experiment(
    model=model,
    L_values=[16, 32],
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
    energy_recompute_stride=100,
    energy_drift_tolerance_per_site=1.0e-10,
    store_primary_histories=False,
    observable_n_blocks=20,
    output_dir="outputs/xy_cpu",
)
```

## Parameter Notes

- Ising model parameters: `J`, `h`, `ordered_start`, `output_prefix`.
- XY model parameters: `J`, `theta_step`, `ordered_start`, `output_prefix`.
- TBG model parameters: `K`, `phase_step`, `amplitude_step`, `ordered_start`,
  `output_prefix`.
- Spin frozen model parameters: `K`, `field_step`, `ordered_start`,
  `output_prefix`.
- GPU-only run parameter: `threads_per_block`.
- `record_stride` controls how often walker labels, swap diagnostics, and local
  acceptance diagnostics are recorded.
- `derived_observable_stride` controls expensive derived measurements such as
  Ising magnetization moments and XY helicity measurements.
- `store_primary_histories=True` saves raw observable time series for
  autocorrelation plots, but makes output files much larger.
- `observable_n_blocks` controls block averages used for jackknife errors.
- `energy_recompute_stride=0` disables drift recomputation checks.
- Output folders under `outputs/` are ignored by Git.
