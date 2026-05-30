from __future__ import annotations
from typing import Any
import numpy as np

from analysis.bkt import estimate_bkt_intersection
from analysis.diagnostics import compute_run_diagnostics
from analysis.helicity_observables import compute_helicity_observables
from analysis.statistics import jackknife_blocks, jackknife_from_block_means
from analysis.thermodynamics import compute_thermodynamics
from analysis.z2_observables import compute_z2_observables

ArrayLike = Any

def has_keys(data: dict[str, Any], keys: list[str]) -> bool:
    """
    Return True if all keys are present in data.
    """
    return all(key in data for key in keys)


def has_nonempty_block_keys(data: dict[str, Any], keys: list[str]) -> bool:
    """
    Return True if all keys exist and contain at least one block column.
    """
    if not has_keys(data, keys):
        return False
    for key in keys:
        arr = np.asarray(data[key])
        if arr.ndim != 2 or arr.shape[1] <= 0:
            return False
    return True


def has_nonempty_block_values(*values: Any) -> bool:
    """
    Return True if all values look like non-empty block matrices.
    """
    for value in values:
        arr = np.asarray(value)
        if arr.ndim != 2 or arr.shape[1] <= 0:
            return False
    return True

def compute_extra_block_observable(
    block_means: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute mean/error for a generic observable stored as block means.
    """
    blocks = np.asarray(block_means, dtype=np.float64)
    if blocks.ndim != 2:
        raise ValueError(
            "block_means must have shape (n_temps, n_blocks). "
            f"Got {blocks.shape}."
        )
    n_temps = blocks.shape[0]
    mean = np.full(n_temps, np.nan)
    err = np.full(n_temps, np.nan)
    for r in range(n_temps):
        mean[r], err[r] = jackknife_from_block_means(blocks[r])
    return mean, err

def compute_extra_observables(
    data: dict[str, Any],
    *,
    extra_observable_specs: dict[str, dict[str, str]] | None = None,
) -> dict[str, np.ndarray]:
    """
    Compute generic extra observables from block means.
    """
    if extra_observable_specs is None:
        return {}
    obs: dict[str, np.ndarray] = {}
    for name, spec in extra_observable_specs.items():
        block_key = spec["block_key"]
        if block_key not in data:
            continue
        mean, err = compute_extra_block_observable(data[block_key])
        obs[name] = mean
        obs[f"{name}_err"] = err
    return obs


def _first_available(data: dict[str, Any], *keys: str) -> Any | None:
    """
    Return the first present value from a list of possible key names.
    """
    for key in keys:
        if key in data:
            return data[key]
    return None


def compute_helicity_from_history(
    helicities: ArrayLike,
    *,
    n_temps: int,
    n_bins: int = 20,
) -> dict[str, np.ndarray]:
    """
    Compute Y and Y_err from saved direct helicity histories.
    """
    values = np.asarray(helicities, dtype=np.float64)
    if values.size == 0:
        return {}
    if values.ndim != 2 or values.shape[0] != n_temps:
        raise ValueError(
            "helicities must have shape (n_temps, n_measurements). "
            f"Got {values.shape}."
        )
    Y = np.full(n_temps, np.nan)
    Y_err = np.full(n_temps, np.nan)
    for r in range(n_temps):
        Y[r], Y_err[r] = jackknife_blocks(values[r], n_blocks=n_bins)
    return {
        "Y": Y,
        "Y_err": Y_err,
    }


def analyze_run(
    data: dict[str, Any],
    *,
    L: int | None = None,
    temps: ArrayLike | None = None,
    energy_per_site: bool = False,
    order_parameter_per_site: bool = False,
    record_stride: int = 1,
    autocorrelation_keys: list[str] | None = None,
    autocorrelation_sample_strides: dict[str, int] | None = None,
    tagged_autocorrelation_keys: list[str] | None = None,
    tagged_autocorrelation_sample_strides: dict[str, int] | None = None,
    extra_observable_specs: dict[str, dict[str, str]] | None = None,
    helicity_history_n_bins: int = 20,
    estimate_bkt: bool = True,
    weber_minnhagen_C: float | None = None,
    bkt_n_bootstrap: int = 2000,
    bkt_rng_seed: int = 12345,
) -> dict[str, Any]:
    """
    Analyze one simulation run.

    This function computes whatever is available from the saved arrays.
    """
    obs: dict[str, Any] = {}
    if temps is None:
        if "temps" not in data:
            raise ValueError("temps was not provided and data['temps'] is missing.")
        temps = data["temps"]
    temps = np.asarray(temps, dtype=np.float64)
    if L is None:
        if "L" not in data:
            raise ValueError("L was not provided and data['L'] is missing.")
        L = int(np.asarray(data["L"]).item())

    L = int(L)
    thermodynamic_keys = [
        "energy_block_means",
        "energy2_block_means",
    ]
    if has_nonempty_block_keys(data, thermodynamic_keys):
        obs.update(
            compute_thermodynamics(
                temps=temps,
                L=L,
                energy_block_means=data["energy_block_means"],
                energy2_block_means=data["energy2_block_means"],
                energy_per_site=energy_per_site,
            )
        )
    order_abs_blocks = _first_available(
        data,
        "order_abs_block_means",
        "mag_abs_block_means",
    )
    order2_blocks = _first_available(
        data,
        "order2_block_means",
        "mag2_block_means",
    )
    order4_blocks = _first_available(
        data,
        "order4_block_means",
        "mag4_block_means",
    )
    if (
        order_abs_blocks is not None
        and order2_blocks is not None
        and order4_blocks is not None
        and has_nonempty_block_values(
            order_abs_blocks,
            order2_blocks,
            order4_blocks,
        )
    ):
        obs.update(
            compute_z2_observables(
                temps=temps,
                L=L,
                order_abs_block_means=order_abs_blocks,
                order2_block_means=order2_blocks,
                order4_block_means=order4_blocks,
                order_parameter_per_site=order_parameter_per_site,
            )
        )
    helicity_keys = [
        "helicity_Kx_block_means",
        "helicity_Ix_block_means",
        "helicity_Ix2_block_means",
        "helicity_Ky_block_means",
        "helicity_Iy_block_means",
        "helicity_Iy2_block_means",
    ]
    if has_nonempty_block_keys(data, helicity_keys):
        obs.update(
            compute_helicity_observables(
                temps=temps,
                L=L,
                helicity_Kx_block_means=data["helicity_Kx_block_means"],
                helicity_Ix_block_means=data["helicity_Ix_block_means"],
                helicity_Ix2_block_means=data["helicity_Ix2_block_means"],
                helicity_Ky_block_means=data["helicity_Ky_block_means"],
                helicity_Iy_block_means=data["helicity_Iy_block_means"],
                helicity_Iy2_block_means=data["helicity_Iy2_block_means"],
            )
        )
    elif "helicities" in data:
        obs.update(
            compute_helicity_from_history(
                data["helicities"],
                n_temps=temps.size,
                n_bins=helicity_history_n_bins,
            )
        )
    if estimate_bkt and "Y" in obs:
        obs["bkt_intersection"] = estimate_bkt_intersection(
            temps=temps,
            Y=obs["Y"],
            Y_err=obs.get("Y_err"),
            L=L,
            weber_minnhagen_C=weber_minnhagen_C,
            n_bootstrap=bkt_n_bootstrap,
            rng_seed=bkt_rng_seed,
        )
    obs.update(
        compute_extra_observables(
            data,
            extra_observable_specs=extra_observable_specs,
        )
    )
    obs.update(
        compute_run_diagnostics(
            data,
            record_stride=record_stride,
            autocorrelation_keys=autocorrelation_keys,
            autocorrelation_sample_strides=autocorrelation_sample_strides,
            tagged_autocorrelation_keys=tagged_autocorrelation_keys,
            tagged_autocorrelation_sample_strides=(
                tagged_autocorrelation_sample_strides
            ),
        )
    )
    obs["temps"] = temps
    obs["L"] = np.int32(L)
    return obs
