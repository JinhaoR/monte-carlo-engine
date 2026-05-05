from __future__ import annotations
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Protocol

import numpy as np
from numba import cuda, float32
from numba.cuda.random import (
    create_xoroshiro128p_states,
    xoroshiro128p_uniform_float32,
)

from physics_model import ChiralU1Z2Model

OUTPUT_SCHEMA_VERSION = 1

class PhysicsRuntime(Protocol):
    """Model runtime contract used by the parallel-tempering driver."""

    energy_drift_last: np.ndarray
    energy_drift_max: np.ndarray
    energy_recompute_checks: np.ndarray
    energy_recompute_corrections: np.ndarray

    @property
    def energy_by_walker(self) -> Any:
        ...

    def maybe_recompute_energy(
        self,
        sweeps_completed: int,
        recompute_stride: int,
        tolerance_per_site: np.float32,
    ) -> None:
        ...

    def allocate_measurement_storage(
        self,
        n_meas: int,
        n_derived_meas: int,
        store_primary_histories: bool,
        observable_n_blocks: int,
    ) -> None:
        ...

    def sweep(self, betas_by_walker: Any, rng_states_updates: Any) -> None:
        ...

    def record_primary_observables(self, walker_of_slot: Any, col: int) -> None:
        ...

    def record_derived_observables(
        self, betas_by_walker: Any, walker_of_slot: Any, col: int
    ) -> None:
        ...

    def copy_measurements_to_host(self) -> dict[str, np.ndarray]:
        ...

    def sync_energy_drift_stats_from_gpu(self) -> dict[str, np.ndarray]:
        ...


class PhysicsModel(Protocol):
    """Host-side physics contract used by the parallel-tempering driver."""

    def max_threads_per_block(self) -> int:
        ...

    def metadata(self) -> dict[str, Any]:
        ...

    def create_runtime(
        self,
        *,
        L: int,
        R: int,
        rng: np.random.Generator,
        theta_step: float,
        threads_per_block: int,
        full_site_blocks: int,
        half_sweep_blocks_per_walker: int,
        slot_blocks: int,
        full_lattice_blocks_per_walker: int,
        inv_N: np.float32,
    ) -> PhysicsRuntime:
        ...


def _resolve_physics_model(
    model: PhysicsModel | None,
    J: float,
    a: float,
) -> PhysicsModel:
    """Create or validate the model object used by a simulation run."""
    if model is None:
        return ChiralU1Z2Model(J=J, a=a)

    model_J = getattr(model, "J", None)
    if (
        model_J is not None
        and not math.isclose(float(J), 1.0)
        and not math.isclose(float(J), float(model_J))
    ):
        raise ValueError("Specify J either directly or through model, not both.")
    model_a = getattr(model, "a", None)
    if model_a is None:
        if not math.isclose(float(a), 1.0):
            raise ValueError("Specify a through a model that exposes an 'a' attribute.")
    elif not math.isclose(float(a), 1.0) and not math.isclose(
        float(a), float(model_a)
    ):
        raise ValueError("Specify a either directly or through model, not both.")
    return model


def _json_default(value: Any) -> Any:
    """Convert NumPy/path values that appear in run metadata to JSON."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def _model_metadata(model: PhysicsModel) -> dict[str, Any]:
    """Return a plain metadata dictionary from a physics model."""
    return dict(model.metadata())


def _safe_output_token(value: Any, default: str = "simulation") -> str:
    """Sanitize a model/file stem for portable output filenames."""
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip().lower())
    token = token.strip("._-")
    return token or default


def _model_output_prefix(model: PhysicsModel) -> str:
    """Choose the filename prefix for saved runs."""
    metadata = _model_metadata(model)
    prefix = (
        metadata.get("output_prefix")
        or metadata.get("file_prefix")
        or metadata.get("model_name")
        or model.__class__.__name__
    )
    return _safe_output_token(prefix)


def _run_output_filename(output_prefix: str, L: int) -> str:
    return f"{_safe_output_token(output_prefix)}_L{int(L)}.npz"


def _trapezoid(y, x) -> float:
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:
        integrate = np.trapz
    return float(integrate(y, x))


# ============================================================
# Utility: ladder construction
# ============================================================


def make_temperature_ladder(
    T_min,
    T_max,
    n_T,
    method="beta",
    dense_near_tc=False,
    Tc=1.0,
    tc_window=0.05,
    tc_fraction=0.50,
):
    """
    Build a monotone temperature ladder, optionally denser near Tc.

    In dense mode this uses a smooth density profile rather than stitching
    together piecewise-uniform segments. That avoids abrupt spacing changes at
    the edges of the dense region, which otherwise show up as swap-acceptance
    cliffs at the ladder-window boundaries.
    """
    T_min = float(T_min)
    T_max = float(T_max)
    n_T = int(n_T)
    method = str(method).lower()

    if n_T < 2:
        return np.array([T_min], dtype=np.float32)

    if not dense_near_tc:
        if method == "beta":
            betas = np.linspace(1.0 / T_min, 1.0 / T_max, n_T)
            return (1.0 / betas).astype(np.float32)
        return np.linspace(T_min, T_max, n_T, dtype=np.float32)

    if method not in {"beta", "linear", "temp", "temperature"}:
        raise ValueError(f"Unsupported ladder method '{method}'.")

    if tc_window <= 0.0:
        raise ValueError("tc_window must be positive when dense_near_tc=True.")
    if not (0.0 < tc_fraction < 1.0):
        raise ValueError("tc_fraction must lie strictly between 0 and 1.")

    T_lo = max(T_min, float(Tc) - float(tc_window) / 2.0)
    T_hi = min(T_max, float(Tc) + float(tc_window) / 2.0)
    if T_hi <= T_lo:
        raise ValueError("Dense Tc window does not overlap the temperature range.")

    if method == "beta":
        coord_min = 1.0 / T_max
        coord_max = 1.0 / T_min

        def temp_to_coord(temp):
            return 1.0 / temp

        def coord_to_temp(coord):
            return 1.0 / coord
    else:
        coord_min = T_min
        coord_max = T_max

        def temp_to_coord(temp):
            return temp

        def coord_to_temp(coord):
            return coord

    coord_lo, coord_hi = sorted((temp_to_coord(T_lo), temp_to_coord(T_hi)))
    coord_center = temp_to_coord(min(max(float(Tc), T_min), T_max))
    coord_span = coord_max - coord_min
    window_span = coord_hi - coord_lo
    if coord_span <= 0.0 or window_span <= 0.0:
        raise ValueError("Degenerate temperature ladder span.")

    n_grid = max(4097, 64 * n_T + 1)
    coord_grid = np.linspace(coord_min, coord_max, n_grid, dtype=np.float32)
    shoulder = max(
        window_span / 8.0,
        coord_span / max(8 * max(n_T - 1, 1), 128),
    )
    profile = 0.5 * (
        np.tanh((coord_grid - coord_lo) / shoulder)
        - np.tanh((coord_grid - coord_hi) / shoulder)
    )
    hard_window = ((coord_grid >= coord_lo) & (coord_grid <= coord_hi)).astype(np.float32)

    base_total_mass = coord_span
    base_window_mass = window_span
    profile_total_mass = _trapezoid(profile, coord_grid)
    profile_window_mass = _trapezoid(profile * hard_window, coord_grid)

    def window_fraction(amplitude: float) -> float:
        numerator = base_window_mass + amplitude * profile_window_mass
        denominator = base_total_mass + amplitude * profile_total_mass
        return numerator / denominator

    base_fraction = base_window_mass / base_total_mass
    target_fraction = max(base_fraction, float(tc_fraction))
    amplitude = 0.0

    if target_fraction > base_fraction + 1e-12 and profile_total_mass > 0.0:
        amp_lo = 0.0
        amp_hi = 1.0
        while window_fraction(amp_hi) < target_fraction and amp_hi < 1.0e12:
            amp_hi *= 2.0

        if window_fraction(amp_hi) >= target_fraction:
            for _ in range(80):
                amp_mid = 0.5 * (amp_lo + amp_hi)
                if window_fraction(amp_mid) < target_fraction:
                    amp_lo = amp_mid
                else:
                    amp_hi = amp_mid
            amplitude = amp_hi
        else:
            amplitude = amp_hi

    density = 1.0 + amplitude * profile
    cdf = np.zeros_like(coord_grid)
    cdf[1:] = np.cumsum(
        0.5 * (density[1:] + density[:-1]) * np.diff(coord_grid)
    )
    total_mass = cdf[-1]
    if total_mass <= 0.0:
        raise RuntimeError("Temperature ladder density integration failed.")
    cdf /= total_mass

    coords = np.interp(np.linspace(0.0, 1.0, n_T), cdf, coord_grid)
    temps = np.sort(coord_to_temp(coords).astype(np.float32))
    temps[0] = T_min
    temps[-1] = T_max
    return temps


def temperature_ladder_diagnostics(temps: np.ndarray) -> dict[str, float]:
    """Summarize adjacent temperature spacing for quick PT checks."""
    temps = np.asarray(temps, dtype=np.float32)
    dtemps = np.diff(temps)
    dtemps = dtemps[dtemps > 0.0]

    if dtemps.size == 0:
        return {
            "min_delta_T": 0.0,
            "max_delta_T": 0.0,
            "delta_T_gap_ratio": 0.0,
            "max_adjacent_delta_ratio": 0.0,
        }

    min_delta_T = float(np.min(dtemps))
    max_delta_T = float(np.max(dtemps))
    if dtemps.size > 1:
        local_ratios = np.maximum(dtemps[1:] / dtemps[:-1], dtemps[:-1] / dtemps[1:])
        max_adjacent_delta_ratio = float(np.max(local_ratios))
    else:
        max_adjacent_delta_ratio = 1.0
    return {
        "min_delta_T": min_delta_T,
        "max_delta_T": max_delta_T,
        "delta_T_gap_ratio": (
            float(max_delta_T / min_delta_T) if min_delta_T > 0.0 else float("inf")
        ),
        "max_adjacent_delta_ratio": max_adjacent_delta_ratio,
    }


# ============================================================
# CUDA kernels: parallel-tempering mechanics
# ============================================================


@cuda.jit
def parallel_tempering_swap_kernel(
    E_by_walker,
    betas,
    walker_of_slot,
    slot_of_walker,
    betas_by_walker,
    rng_states,
    parity,
    swap_attempts,
    swap_acceptance,
):
    """Attempt disjoint neighboring replica exchanges directly on the GPU."""
    pair_idx = cuda.grid(1)
    R = walker_of_slot.shape[0]
    slot = parity + 2 * pair_idx

    if slot + 1 >= R:
        return

    wi = walker_of_slot[slot]
    wj = walker_of_slot[slot + 1]

    beta_i = betas[slot]
    beta_j = betas[slot + 1]
    d = (beta_i - beta_j) * (E_by_walker[wj] - E_by_walker[wi])

    swap_attempts[slot] += 1

    accept = d <= float32(0.0)
    if not accept:
        acc = xoroshiro128p_uniform_float32(rng_states, pair_idx)
        accept = acc < float32(math.exp(-d))

    if accept:
        walker_of_slot[slot] = wj
        walker_of_slot[slot + 1] = wi
        slot_of_walker[wi] = slot + 1
        slot_of_walker[wj] = slot
        betas_by_walker[wi] = beta_j
        betas_by_walker[wj] = beta_i
        swap_acceptance[slot] += 1


@cuda.jit
def record_positions_kernel(slot_of_walker, out, row):
    """Record slot positions for all walkers into a device-side history."""
    walker = cuda.grid(1)
    if walker < slot_of_walker.shape[0]:
        out[row, walker] = slot_of_walker[walker]


# ============================================================
# GPU simulation class
# ============================================================

class ParallelTemperingGPU:
    def __init__(
        self,
        L: int,
        temps: np.ndarray,
        n_equil_sweeps: int,
        n_measure_sweeps: int,
        sweeps_between_swaps: int = 1,
        record_stride: int = 10,
        J: float = 1.0,
        a: float = 1.0,
        theta_step: float = math.pi / 2.0,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
        threads_per_block: int = 128,
        model: PhysicsModel | None = None,
        energy_recompute_stride: int = 0,
        energy_drift_tolerance_per_site: float | None = 1.0e-5,
    ):
        if not cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. Install a working NVIDIA CUDA stack and "
                "the numba-cuda package, then try again."
            )

        self.L = int(L)
        if self.L % 2 != 0:
            raise ValueError(
                "Checkerboard updates with periodic boundaries require even L."
            )

        self.temps = np.asarray(temps, dtype=np.float32)
        if self.temps.ndim != 1:
            raise ValueError("temps must be a one-dimensional array.")
        if not np.all(np.isfinite(self.temps)) or np.any(self.temps <= 0.0):
            raise ValueError("temps must contain finite positive temperatures.")
        if np.any(np.diff(self.temps) <= 0.0):
            raise ValueError("temps must be strictly increasing.")
        self.betas = np.float32(1.0) / self.temps
        self.n_equil_sweeps = int(n_equil_sweeps)
        self.n_measure_sweeps = int(n_measure_sweeps)
        self.sweeps_between_swaps = int(sweeps_between_swaps)
        self.record_stride = int(record_stride)
        if self.sweeps_between_swaps <= 0:
            raise ValueError("sweeps_between_swaps must be positive.")
        if self.record_stride <= 0:
            raise ValueError("record_stride must be positive.")

        self.model = _resolve_physics_model(model, J, a)

        self.energy_recompute_stride = int(energy_recompute_stride)
        if self.energy_recompute_stride < 0:
            raise ValueError("energy_recompute_stride must be non-negative.")
        self.energy_drift_tolerance_per_site_config = energy_drift_tolerance_per_site
        if energy_drift_tolerance_per_site is None:
            self.energy_drift_tolerance_per_site = np.float32(-1.0)
        else:
            if energy_drift_tolerance_per_site < 0.0:
                raise ValueError(
                    "energy_drift_tolerance_per_site must be non-negative or None."
                )
            self.energy_drift_tolerance_per_site = np.float32(
                energy_drift_tolerance_per_site
            )

        self.threads_per_block = int(threads_per_block)
        model_thread_limit = int(self.model.max_threads_per_block())
        if self.threads_per_block > model_thread_limit:
            raise ValueError(
                f"threads_per_block={self.threads_per_block} exceeds the "
                f"model reduction limit of {model_thread_limit}."
            )
        if self.threads_per_block & (self.threads_per_block - 1):
            raise ValueError(
                "threads_per_block must be a power of two for the shared-memory "
                "reductions."
            )

        if rng is None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = rng

        self.R = len(self.temps)
        if self.R < 2:
            raise ValueError("At least two temperatures are required for PT.")

        # Full-lattice launch geometry, used by the physics runtime as needed.
        self.total_sites = self.R * self.L * self.L
        self.blocks = (
            self.total_sites + self.threads_per_block - 1
        ) // self.threads_per_block

        # Packed checkerboard launch geometry, also handed to the physics runtime.
        self.sites_per_replica_half = self.L * (self.L // 2)
        self.active_sites = self.R * self.sites_per_replica_half
        self.blocks_half_per_walker = (
            self.sites_per_replica_half + self.threads_per_block - 1
        ) // self.threads_per_block
        self.area_per_replica = self.L * self.L
        self.full_lattice_blocks_per_walker = (
            self.area_per_replica + self.threads_per_block - 1
        ) // self.threads_per_block
        self.slot_blocks = (
            self.R + self.threads_per_block - 1
        ) // self.threads_per_block
        self.max_swap_pairs = max(1, self.R // 2)
        self.swap_pair_blocks = (
            self.max_swap_pairs + self.threads_per_block - 1
        ) // self.threads_per_block

        self.physics_runtime = self.model.create_runtime(
            L=self.L,
            R=self.R,
            rng=self.rng,
            theta_step=theta_step,
            threads_per_block=self.threads_per_block,
            full_site_blocks=self.blocks,
            half_sweep_blocks_per_walker=self.blocks_half_per_walker,
            slot_blocks=self.slot_blocks,
            full_lattice_blocks_per_walker=self.full_lattice_blocks_per_walker,
            inv_N=np.float32(1.0 / float(self.L * self.L)),
        )
        self.d_betas = cuda.to_device(self.betas)
        self.d_betas_by_walker = cuda.to_device(self.betas.copy())
        self.d_walker_of_slot = cuda.to_device(np.arange(self.R, dtype=np.int32))
        self.d_slot_of_walker = cuda.to_device(np.arange(self.R, dtype=np.int32))

        # RNG states: one set for updates, one for swaps.
        self.rng_states_updates = create_xoroshiro128p_states(
            self.active_sites, seed=np.uint64(seed or 0)
        )
        self.rng_states_swaps = create_xoroshiro128p_states(
            self.max_swap_pairs, seed=np.uint64((seed or 0) + 1)
        )

        # PT stats are tracked by temperature slot.
        self.swap_acceptance = np.zeros(self.R - 1, dtype=np.int64)
        self.swap_attempts = np.zeros(self.R - 1, dtype=np.int64)
        self.d_swap_acceptance = cuda.to_device(self.swap_acceptance)
        self.d_swap_attempts = cuda.to_device(self.swap_attempts)
        self._swap_parity = 0

        self.label_positions = None
        self._label_record_times: list[int] = []
        self._label_record_count = 0
        self.d_label_positions = None

        self.energies = None
        self.mags = None
        self.helicities = None
        self.energy_drift_last = self.physics_runtime.energy_drift_last
        self.energy_drift_max = self.physics_runtime.energy_drift_max
        self.energy_recompute_checks = self.physics_runtime.energy_recompute_checks
        self.energy_recompute_corrections = (
            self.physics_runtime.energy_recompute_corrections
        )
        self.derived_observable_measure_sweeps = np.empty(0, dtype=np.int32)

    def _maybe_recompute_energy(self, sweeps_completed: int):
        """Periodically compare running energy against an exact GPU recomputation."""
        self.physics_runtime.maybe_recompute_energy(
            sweeps_completed,
            self.energy_recompute_stride,
            self.energy_drift_tolerance_per_site,
        )

    def _sync_swap_stats_from_gpu(self):
        self.d_swap_acceptance.copy_to_host(self.swap_acceptance)
        self.d_swap_attempts.copy_to_host(self.swap_attempts)

    def _sync_energy_drift_stats_from_gpu(self):
        stats = self.physics_runtime.sync_energy_drift_stats_from_gpu()
        self.energy_drift_last = stats["energy_drift_last"]
        self.energy_drift_max = stats["energy_drift_max"]
        self.energy_recompute_checks = stats["energy_recompute_checks"]
        self.energy_recompute_corrections = stats[
            "energy_recompute_corrections"
        ]

    def _allocate_measurement_storage(
        self,
        n_meas: int,
        n_derived_meas: int,
        store_primary_histories: bool,
        observable_n_blocks: int,
    ):
        self.physics_runtime.allocate_measurement_storage(
            n_meas,
            n_derived_meas,
            store_primary_histories,
            observable_n_blocks,
        )

    def _advance_model_state(self):
        self.physics_runtime.sweep(
            self.d_betas_by_walker,
            self.rng_states_updates,
        )

    def _allocate_label_storage(self, record_during_equil: bool):
        equil_swaps = self.n_equil_sweeps // self.sweeps_between_swaps
        meas_swaps = self.n_measure_sweeps // self.sweeps_between_swaps
        total_swaps = equil_swaps + meas_swaps

        if record_during_equil:
            n_records = total_swaps // self.record_stride
        else:
            n_records = (
                total_swaps // self.record_stride
                - equil_swaps // self.record_stride
            )
        n_records = max(0, int(n_records))

        self._label_record_times = []
        self._label_record_count = 0
        self.d_label_positions = (
            cuda.device_array((n_records, self.R), dtype=np.int32)
            if n_records > 0
            else None
        )

    def _record_observables_to_output(self, col: int):
        self.physics_runtime.record_primary_observables(
            self.d_walker_of_slot,
            col,
        )

    def _attempt_swaps(self):
        """
        Attempt swaps between neighboring temperature slots.
        """
        parallel_tempering_swap_kernel[self.swap_pair_blocks, self.threads_per_block](
            self.physics_runtime.energy_by_walker,
            self.d_betas,
            self.d_walker_of_slot,
            self.d_slot_of_walker,
            self.d_betas_by_walker,
            self.rng_states_swaps,
            self._swap_parity,
            self.d_swap_attempts,
            self.d_swap_acceptance,
        )
        self._swap_parity ^= 1

    def _record_positions(self, t_index: int):
        if self.d_label_positions is None:
            return
        if self._label_record_count >= self.d_label_positions.shape[0]:
            return

        record_positions_kernel[self.slot_blocks, self.threads_per_block](
            self.d_slot_of_walker,
            self.d_label_positions,
            self._label_record_count,
        )
        self._label_record_times.append(int(t_index))
        self._label_record_count += 1

    def _compute_round_trip_stats(
        self, label_pos_arr: np.ndarray, record_times: list[int]
    ):
        rt_state = np.zeros(self.R, dtype=np.int8)
        rt_counts = np.zeros(self.R, dtype=np.int64)
        commute_counts = np.zeros(self.R, dtype=np.int64)
        rt_last_t = -np.ones(self.R, dtype=np.int64)
        rt_durations: list[int] = []
        hit_low = np.zeros(self.R, dtype=np.bool_)
        hit_high = np.zeros(self.R, dtype=np.bool_)
        low = 0
        high = self.R - 1

        for rec_idx, t_index in enumerate(record_times):
            pos = label_pos_arr[rec_idx]
            for lab in range(self.R):
                p = int(pos[lab])
                st = int(rt_state[lab])

                if p == low:
                    hit_low[lab] = True
                elif p == high:
                    hit_high[lab] = True

                if st == 0:
                    if p == low:
                        rt_state[lab] = 1
                        rt_last_t[lab] = t_index
                    elif p == high:
                        rt_state[lab] = 2
                        rt_last_t[lab] = t_index
                    continue

                if st == 1:
                    if p == high:
                        commute_counts[lab] += 1
                        rt_state[lab] = 3
                    continue

                if st == 2:
                    if p == low:
                        commute_counts[lab] += 1
                        rt_state[lab] = 4
                    continue

                if st == 3:
                    if p == low and rt_last_t[lab] >= 0:
                        rt_counts[lab] += 1
                        rt_durations.append(int(t_index - rt_last_t[lab]))
                        rt_last_t[lab] = t_index
                        rt_state[lab] = 1
                    continue

                if st == 4:
                    if p == high and rt_last_t[lab] >= 0:
                        rt_counts[lab] += 1
                        rt_durations.append(int(t_index - rt_last_t[lab]))
                        rt_last_t[lab] = t_index
                        rt_state[lab] = 2

        return (
            rt_counts,
            np.asarray(rt_durations, dtype=np.int64),
            commute_counts,
            hit_low,
            hit_high,
            hit_low & hit_high,
        )

    def _record_derived_observables_to_output(self, col: int):
        self.physics_runtime.record_derived_observables(
            self.d_betas_by_walker,
            self.d_walker_of_slot,
            col,
        )

    def run(
        self,
        record_during_equil: bool = False,
        derived_observable_stride: int = 5,
        store_primary_histories: bool = True,
        observable_n_blocks: int = 20,
    ):
        derived_observable_stride = int(derived_observable_stride)
        if derived_observable_stride <= 0:
            raise ValueError("derived_observable_stride must be positive.")
        observable_n_blocks = int(observable_n_blocks)
        if observable_n_blocks <= 0:
            raise ValueError("observable_n_blocks must be positive.")

        n_meas = self.n_measure_sweeps
        self.derived_observable_measure_sweeps = np.arange(
            0, n_meas, derived_observable_stride, dtype=np.int32
        )
        n_derived_meas = int(self.derived_observable_measure_sweeps.size)

        self._allocate_measurement_storage(
            n_meas,
            n_derived_meas,
            bool(store_primary_histories),
            observable_n_blocks,
        )
        self._allocate_label_storage(record_during_equil)

        t_counter = 0
        derived_col = 0

        for sweep in range(self.n_equil_sweeps):
            self._advance_model_state()
            self._maybe_recompute_energy(sweep + 1)
            if (sweep + 1) % self.sweeps_between_swaps == 0:
                self._attempt_swaps()
                t_counter += 1
                if record_during_equil and (t_counter % self.record_stride == 0):
                    self._record_positions(t_counter)

        for meas_idx in range(n_meas):
            self._advance_model_state()
            self._maybe_recompute_energy(self.n_equil_sweeps + meas_idx + 1)
            if (meas_idx + 1) % self.sweeps_between_swaps == 0:
                self._attempt_swaps()
                t_counter += 1
                if t_counter % self.record_stride == 0:
                    self._record_positions(t_counter)

            self._record_observables_to_output(meas_idx)

            if meas_idx % derived_observable_stride == 0:
                self._record_derived_observables_to_output(derived_col)
                derived_col += 1

        measurements = self.physics_runtime.copy_measurements_to_host()
        self.energies = measurements["energies"]
        self.mags = measurements["mags"]
        self.helicities = measurements["helicities"]
        self._sync_swap_stats_from_gpu()
        self._sync_energy_drift_stats_from_gpu()

        label_pos_arr = (
            self.d_label_positions.copy_to_host()
            if self.d_label_positions is not None
            else np.zeros((0, self.R), dtype=np.int32)
        )
        self.label_positions = label_pos_arr
        (
            rt_counts,
            rt_durations,
            commute_counts,
            hit_low,
            hit_high,
            hit_both_edges,
        ) = self._compute_round_trip_stats(
            label_pos_arr, self._label_record_times
        )

        return {
            **measurements,
            "energies": self.energies,
            "mags": self.mags,
            "helicities": self.helicities,
            "derived_observable_measure_sweeps": (
                self.derived_observable_measure_sweeps
            ),
            "temps": self.temps,
            "betas": self.betas,
            "swap_acceptance": self.swap_acceptance,
            "swap_attempts": self.swap_attempts,
            "energy_drift_last": self.energy_drift_last,
            "energy_drift_max": self.energy_drift_max,
            "energy_recompute_checks": self.energy_recompute_checks,
            "energy_recompute_corrections": self.energy_recompute_corrections,
            "label_positions": label_pos_arr,
            "round_trip_counts": rt_counts,
            "round_trip_durations": rt_durations,
            "commute_counts": commute_counts,
            "hit_low": hit_low,
            "hit_high": hit_high,
            "hit_both_edges": hit_both_edges,
        }


# ============================================================
# Data production only
# ============================================================


def _mean_blocks(arr: np.ndarray, n_blocks: int) -> tuple[np.ndarray, int]:
    """Return contiguous block means along the measurement axis."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("Expected a 2D array of shape (R, n_measurements).")

    n_meas = arr.shape[1]
    if n_meas <= 0:
        return np.empty((arr.shape[0], 0), dtype=np.float32), 0

    requested_blocks = max(1, int(n_blocks))
    n_blocks = min(requested_blocks, n_meas // 2)
    if n_blocks < 2:
        n_blocks = 1
    block_size = n_meas // n_blocks
    if block_size <= 0:
        return np.empty((arr.shape[0], 0), dtype=np.float32), 0

    n_use = n_blocks * block_size
    blocks = arr[:, :n_use].reshape(arr.shape[0], n_blocks, block_size)
    return np.mean(blocks, axis=2), block_size


def _build_compact_measurement_stats(
    energies: np.ndarray,
    mags: np.ndarray,
    observable_n_blocks: int,
) -> dict[str, np.ndarray]:
    """Build compact block-moment summaries for plotting and error analysis."""
    energies = np.asarray(energies, dtype=np.float32)
    mags = np.asarray(mags, dtype=np.float32)

    energy_block_means, block_size = _mean_blocks(energies, observable_n_blocks)
    if block_size <= 0:
        empty = np.empty((energies.shape[0], 0), dtype=np.float32)
        return {
            "energy_block_means": empty,
            "energy2_block_means": empty,
            "mag_abs_block_means": empty,
            "mag2_block_means": empty,
            "mag4_block_means": empty,
            "observable_block_size": np.int32(0),
        }

    n_blocks = energy_block_means.shape[1]
    n_use = n_blocks * block_size
    energy2_block_means, _ = _mean_blocks(energies[:, :n_use] ** 2, n_blocks)
    mag_abs_block_means, _ = _mean_blocks(np.abs(mags[:, :n_use]), n_blocks)
    mag2_block_means, _ = _mean_blocks(mags[:, :n_use] ** 2, n_blocks)
    mag4_block_means, _ = _mean_blocks(mags[:, :n_use] ** 4, n_blocks)

    return {
        "energy_block_means": energy_block_means,
        "energy2_block_means": energy2_block_means,
        "mag_abs_block_means": mag_abs_block_means,
        "mag2_block_means": mag2_block_means,
        "mag4_block_means": mag4_block_means,
        "observable_block_size": np.int32(block_size),
    }


def save_run_npz(
    out_path: Path,
    L: int,
    res: dict[str, np.ndarray],
    params: dict[str, Any],
    save_full_measurement_histories: bool = False,
    observable_n_blocks: int = 20,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    compact_keys = (
        "energy_block_means",
        "energy2_block_means",
        "mag_abs_block_means",
        "mag2_block_means",
        "mag4_block_means",
        "observable_block_size",
    )
    if all(key in res and res[key] is not None for key in compact_keys):
        compact = {key: res[key] for key in compact_keys}
    else:
        compact = _build_compact_measurement_stats(
            res["energies"],
            res["mags"],
            observable_n_blocks=observable_n_blocks,
        )
    energies_to_save = (
        np.asarray(res["energies"], dtype=np.float32)
        if save_full_measurement_histories
        else np.empty((len(res["temps"]), 0), dtype=np.float32)
    )
    mags_to_save = (
        np.asarray(res["mags"], dtype=np.float32)
        if save_full_measurement_histories
        else np.empty((len(res["temps"]), 0), dtype=np.float32)
    )
    derived_observable_measure_sweeps = np.asarray(
        res["derived_observable_measure_sweeps"], dtype=np.int32
    )
    np.savez_compressed(
        out_path,
        schema_version=np.int32(OUTPUT_SCHEMA_VERSION),
        L=np.int32(L),
        temps=res["temps"],
        betas=res["betas"],
        energies=energies_to_save,
        mags=mags_to_save,
        energy_block_means=compact["energy_block_means"],
        energy2_block_means=compact["energy2_block_means"],
        mag_abs_block_means=compact["mag_abs_block_means"],
        mag2_block_means=compact["mag2_block_means"],
        mag4_block_means=compact["mag4_block_means"],
        observable_block_size=compact["observable_block_size"],
        helicities=res["helicities"],
        derived_observable_measure_sweeps=derived_observable_measure_sweeps,
        helicity_measure_sweeps=derived_observable_measure_sweeps,
        swap_acceptance=res["swap_acceptance"],
        swap_attempts=res["swap_attempts"],
        energy_drift_last=res.get(
            "energy_drift_last", np.empty(0, dtype=np.float32)
        ),
        energy_drift_max=res.get("energy_drift_max", np.empty(0, dtype=np.float32)),
        energy_recompute_checks=res.get(
            "energy_recompute_checks", np.empty(0, dtype=np.int64)
        ),
        energy_recompute_corrections=res.get(
            "energy_recompute_corrections", np.empty(0, dtype=np.int64)
        ),
        label_positions=res["label_positions"],
        round_trip_counts=res["round_trip_counts"],
        round_trip_durations=res["round_trip_durations"],
        commute_counts=res["commute_counts"],
        hit_low=res["hit_low"],
        hit_high=res["hit_high"],
        hit_both_edges=res["hit_both_edges"],
        params_json=np.array(json.dumps(params, default=_json_default)),
    )


def run_full_experiment_gpu(
    L_values: Iterable[int] | None = None,
    T_min: float = 0.2,
    T_max: float = 2.0,
    n_T: int = 16,
    ladder_method: str = "beta",
    dense_near_tc: bool = False,
    tc_window: float = 0.05,
    tc_fraction: float = 0.50,
    n_equil_sweeps: int = 5000,
    n_measure_sweeps: int = 10000,
    sweeps_between_swaps: int = 1,
    record_stride: int = 10,
    derived_observable_stride: int = 5,
    rng_seed: int = 1234,
    T_focus: float = 1.0,
    J: float = 1.0,
    a: float = 1.0,
    model: PhysicsModel | None = None,
    theta_step: float = math.pi / 2.0,
    save_full_measurement_histories: bool = False,
    observable_n_blocks: int = 20,
    threads_per_block: int = 128,
    energy_recompute_stride: int = 0,
    energy_drift_tolerance_per_site: float | None = 1.0e-5,
    output_dir: str | Path = "gpu_raw_runs",
    output_prefix: str | None = None,
):
    if L_values is None:
        L_values = [16, 32]
    L_values = [int(L) for L in L_values]

    for L in L_values:
        if L % 2 != 0:
            raise ValueError(
                f"L={L} is odd. Checkerboard updates with periodic boundaries require even L."
            )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = _resolve_physics_model(model, J, a)
    model_metadata = _model_metadata(model)
    J = float(getattr(model, "J", J))
    a = float(getattr(model, "a", a))
    output_prefix = (
        _model_output_prefix(model)
        if output_prefix is None
        else _safe_output_token(output_prefix)
    )
    derived_observable_stride = int(derived_observable_stride)
    if derived_observable_stride <= 0:
        raise ValueError("derived_observable_stride must be positive.")

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
    ladder_diag = temperature_ladder_diagnostics(temps)
    print(
        "Temperature ladder diagnostics: "
        f"min dT={ladder_diag['min_delta_T']:.6f}, "
        f"max dT={ladder_diag['max_delta_T']:.6f}, "
        f"gap ratio={ladder_diag['delta_T_gap_ratio']:.3f}, "
        f"max adjacent ratio={ladder_diag['max_adjacent_delta_ratio']:.3f}",
        flush=True,
    )
    if ladder_diag["delta_T_gap_ratio"] > 3.0:
        print(
            "Note: overall ladder spacing varies strongly across the full range. "
            "That is expected for a Tc-focused ladder; the local adjacent-spacing "
            "ratio is the better diagnostic for boundary kinks.",
            flush=True,
        )
    if ladder_diag["max_adjacent_delta_ratio"] > 1.5:
        print(
            "Warning: adjacent ladder spacings change abruptly somewhere on the ladder. "
            "That can produce swap-acceptance cliffs near the transition into the dense region.",
            flush=True,
        )

    manifest = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "output_dir": str(output_dir.resolve()),
        "files": [],
        "output_prefix": output_prefix,
        "temps": temps.tolist(),
        "ladder_diagnostics": ladder_diag,
        "params": {
            "T_min": T_min,
            "T_max": T_max,
            "n_T": n_T,
            "ladder_method": ladder_method,
            "dense_near_tc": dense_near_tc,
            "tc_window": tc_window,
            "tc_fraction": tc_fraction,
            "n_equil_sweeps": n_equil_sweeps,
            "n_measure_sweeps": n_measure_sweeps,
            "sweeps_between_swaps": sweeps_between_swaps,
            "record_stride": record_stride,
            "derived_observable_stride": derived_observable_stride,
            "rng_seed": rng_seed,
            "T_focus": T_focus,
            "J": J,
            "a": a,
            "model": model_metadata,
            "output_prefix": output_prefix,
            "theta_step": theta_step,
            "save_full_measurement_histories": save_full_measurement_histories,
            "observable_n_blocks": observable_n_blocks,
            "threads_per_block": threads_per_block,
            "energy_recompute_stride": energy_recompute_stride,
            "energy_drift_tolerance_per_site": energy_drift_tolerance_per_site,
        },
    }

    for idx, L in enumerate(L_values):
        print(f"Running GPU simulation for L={L} ...", flush=True)
        rng = np.random.default_rng(int(rng_seed) + idx)

        sim = ParallelTemperingGPU(
            L=L,
            temps=temps,
            n_equil_sweeps=n_equil_sweeps,
            n_measure_sweeps=n_measure_sweeps,
            sweeps_between_swaps=sweeps_between_swaps,
            record_stride=record_stride,
            J=J,
            a=a,
            theta_step=theta_step,
            seed=int(rng_seed) + idx,
            rng=rng,
            threads_per_block=threads_per_block,
            model=model,
            energy_recompute_stride=energy_recompute_stride,
            energy_drift_tolerance_per_site=energy_drift_tolerance_per_site,
        )

        res = sim.run(
            record_during_equil=False,
            derived_observable_stride=derived_observable_stride,
            store_primary_histories=save_full_measurement_histories,
            observable_n_blocks=observable_n_blocks,
        )

        params = {
            "L": int(L),
            "T_min": float(T_min),
            "T_max": float(T_max),
            "n_T": int(n_T),
            "ladder_method": str(ladder_method),
            "dense_near_tc": bool(dense_near_tc),
            "tc_window": float(tc_window),
            "tc_fraction": float(tc_fraction),
            "n_equil_sweeps": int(n_equil_sweeps),
            "n_measure_sweeps": int(n_measure_sweeps),
            "sweeps_between_swaps": int(sweeps_between_swaps),
            "record_stride": int(record_stride),
            "derived_observable_stride": int(derived_observable_stride),
            "rng_seed": int(rng_seed) + idx,
            "T_focus": float(T_focus),
            "J": float(J),
            "a": float(a),
            "model": model_metadata,
            "output_prefix": output_prefix,
            "theta_step": float(theta_step),
            "save_full_measurement_histories": bool(save_full_measurement_histories),
            "observable_n_blocks": int(observable_n_blocks),
            "threads_per_block": int(threads_per_block),
            "energy_recompute_stride": int(energy_recompute_stride),
            "energy_drift_tolerance_per_site": (
                None
                if energy_drift_tolerance_per_site is None
                else float(energy_drift_tolerance_per_site)
            ),
        }

        out_file = output_dir / _run_output_filename(output_prefix, L)
        save_run_npz(
            out_file,
            L,
            res,
            params,
            save_full_measurement_histories=save_full_measurement_histories,
            observable_n_blocks=observable_n_blocks,
        )
        manifest["files"].append(str(out_file.name))
        print(f"Saved {out_file}", flush=True)

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(f"Wrote manifest to {manifest_path}")
    print("Done.")


if __name__ == "__main__":
    run_full_experiment_gpu(
        L_values=[32, 64, 128, 256],
        T_min=1.25,
        T_max=1.45,
        n_T=100,
        ladder_method="beta",
        dense_near_tc=True,
        tc_window=0.05,
        tc_fraction=0.5,
        n_equil_sweeps=2_500_000,
        n_measure_sweeps=3_000_000,
        sweeps_between_swaps=10,
        record_stride=10,
        derived_observable_stride=50,
        rng_seed=1234,
        T_focus=1.365,
        J=1.0,
        a=1.0,
        theta_step=1.57079632679,
        threads_per_block=128,
        energy_recompute_stride=10_000,
        energy_drift_tolerance_per_site=1.0e-5,
        output_dir="chiral_data",
    )
