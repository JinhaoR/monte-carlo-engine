from __future__ import annotations

from numba import cuda, float32


@cuda.jit
def correct_energy_drift_kernel(
    E_running,
    E_exact,
    drift_last,
    drift_max,
    recompute_checks,
    recompute_corrections,
    tolerance_per_site,
    inv_N,
):
    r = cuda.grid(1)
    if r >= E_running.shape[0]:
        return

    drift = E_exact[r] - E_running[r]
    if drift < float32(0.0):
        drift = -drift
    drift_last[r] = drift
    if drift > drift_max[r]:
        drift_max[r] = drift
    recompute_checks[r] += 1
    if tolerance_per_site < float32(0.0) or drift * inv_N > tolerance_per_site:
        E_running[r] = E_exact[r]
        recompute_corrections[r] += 1
