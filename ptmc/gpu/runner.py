from __future__ import annotations
from typing import Any
import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states

from ptmc.gpu.interface import BaseModel, validate_pt_model
from ptmc.gpu.kernels import (
    parallel_tempering_swap_kernel,
    record_positions_kernel,
)

class ParallelTemperingGPU:
    """
    Run one GPU parallel tempering simulation.

    This class owns the parallel tempering machinery like
    temperatures, betas, walkers, slots, swap attempts, swap acceptances,
    label tracking, and the simulation loop.

    The model runtime owns the actual physical state and model specific updates.
    """

    def __init__(
        self,
        L: int,
        temps: np.ndarray,
        n_equil_sweeps: int,
        n_measure_sweeps: int,
        *,
        model: BaseModel,
        sweeps_between_swaps: int = 1,
        record_stride: int = 10,
        field_step: float = 0.25,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
        threads_per_block: int = 128,
        energy_recompute_stride: int = 0,
        energy_drift_tolerance_per_site: float | None = 1.0e-5,
    ):
        if not cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. Install a working NVIDIA CUDA stack."
            )
        self.L = int(L)
        if self.L <= 0:
            raise ValueError("L must be positive.")
        self.model = validate_pt_model(model)
        self.model.validate_lattice(self.L)

        self.temps = np.asarray(temps, dtype=np.float32)
        if self.temps.ndim != 1:
            raise ValueError("temps must be a one-dimensional array.")
        if not np.all(np.isfinite(self.temps)):
            raise ValueError("temps must contain only finite values.")
        if np.any(self.temps <= 0.0):
            raise ValueError("temps must contain positive temperatures.")
        if np.any(np.diff(self.temps) <= 0.0):
            raise ValueError("temps must be strictly increasing.")
        self.betas = np.float32(1.0) / self.temps
        self.n_equil_sweeps = int(n_equil_sweeps)
        self.n_measure_sweeps = int(n_measure_sweeps)
        if self.n_equil_sweeps < 0:
            raise ValueError("n_equil_sweeps must be non-negative.")
        if self.n_measure_sweeps <= 0:
            raise ValueError("n_measure_sweeps must be positive.")
        self.sweeps_between_swaps = int(sweeps_between_swaps)
        self.record_stride = int(record_stride)
        if self.sweeps_between_swaps <= 0:
            raise ValueError("sweeps_between_swaps must be positive.")
        if self.record_stride <= 0:
            raise ValueError("record_stride must be positive.")
        self.energy_recompute_stride = int(energy_recompute_stride)
        if self.energy_recompute_stride < 0:
            raise ValueError("energy_recompute_stride must be non-negative.")
        self.energy_drift_tolerance_per_site_config = (
            energy_drift_tolerance_per_site
        )
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
        if self.threads_per_block <= 0:
            raise ValueError("threads_per_block must be positive.")
        model_thread_limit = int(self.model.max_threads_per_block())
        if self.threads_per_block > model_thread_limit:
            raise ValueError(
                f"threads_per_block={self.threads_per_block} exceeds the "
                f"model limit of {model_thread_limit}."
            )
        if self.threads_per_block & (self.threads_per_block - 1):
            raise ValueError("threads_per_block must be a power of two.")
        self.rng = np.random.default_rng(seed) if rng is None else rng
        self.R = len(self.temps)
        if self.R < 2:
            raise ValueError("At least two temperatures are required for PT.")
        self._setup_launch_geometry()
        self._create_model_runtime(field_step=field_step)
        self._allocate_pt_arrays(seed=seed)
        self._allocate_swap_stats()
        self.label_positions: np.ndarray | None = None
        self._label_record_times: list[int] = []
        self._label_record_count = 0
        self.d_label_positions = None
        self.derived_observable_measure_sweeps = np.empty(0, dtype=np.int32)

    def _setup_launch_geometry(self) -> None:
        """
        Compute GPU launch sizes used by the PT runner and passed to the model.
        Some models may use all of these. Some models may ignore some of them.
        """
        self.total_sites = self.R * self.L * self.L
        self.full_site_blocks = (
            self.total_sites + self.threads_per_block - 1
        ) // self.threads_per_block
        model_geometry = self.model.launch_geometry(
            L=self.L,
            R=self.R,
            threads_per_block=self.threads_per_block,
        )
        self.update_sites_per_walker = model_geometry.update_sites_per_walker
        self.update_blocks_per_walker = model_geometry.update_blocks_per_walker
        self.active_sites = model_geometry.update_rng_states
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

    def _create_model_runtime(self, field_step: float) -> None:
        """
        Ask the model to create its live runtime.
        This is where the runner hands general simulation information to the model.
        """
        self.runtime = self.model.create_runtime(
            L=self.L,
            R=self.R,
            rng=self.rng,
            field_step=field_step,
            threads_per_block=self.threads_per_block,
            full_site_blocks=self.full_site_blocks,
            update_blocks_per_walker=self.update_blocks_per_walker,
            slot_blocks=self.slot_blocks,
            full_lattice_blocks_per_walker=self.full_lattice_blocks_per_walker,
            inv_N=np.float32(1.0 / float(self.L * self.L)),
        )

    def _allocate_pt_arrays(self, seed: int | None) -> None:
        """
        Allocate arrays owned by the PT algorithm.
        """
        self.d_betas = cuda.to_device(self.betas)
        self.d_betas_by_walker = cuda.to_device(self.betas.copy())
        self.d_walker_of_slot = cuda.to_device(
            np.arange(self.R, dtype=np.int32)
        )
        self.d_slot_of_walker = cuda.to_device(
            np.arange(self.R, dtype=np.int32)
        )
        base_seed = 0 if seed is None else int(seed)
        self.rng_states_updates = create_xoroshiro128p_states(
            self.active_sites,
            seed=np.uint64(base_seed),
        )
        self.rng_states_swaps = create_xoroshiro128p_states(
            self.max_swap_pairs,
            seed=np.uint64(base_seed + 1),
        )

    def _allocate_swap_stats(self) -> None:
        """
        Allocate swap-attempt and swap-acceptance counters.
        """
        self.swap_acceptance = np.zeros(self.R - 1, dtype=np.int64)
        self.swap_attempts = np.zeros(self.R - 1, dtype=np.int64)
        self.d_swap_acceptance = cuda.to_device(self.swap_acceptance)
        self.d_swap_attempts = cuda.to_device(self.swap_attempts)
        self._swap_parity = 0

    def _maybe_recompute_energy(self, sweeps_completed: int) -> None:
        """
        Ask the model runtime to optionally recompute energy exactly.
        """
        self.runtime.maybe_recompute_energy(
            sweeps_completed,
            self.energy_recompute_stride,
            self.energy_drift_tolerance_per_site,
        )

    def _advance_model_state(self) -> None:
        """
        Ask the model runtime to perform one sweep.
        """
        self.runtime.sweep(
            self.d_betas_by_walker,
            self.rng_states_updates,
            self.d_slot_of_walker,
        )

    def _attempt_swaps(self) -> None:
        """
        Attempt neighboring replica swaps.
        """
        parallel_tempering_swap_kernel[
            self.swap_pair_blocks,
            self.threads_per_block,
        ](
            self.runtime.energy_by_walker,
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

    def _sync_swap_stats_from_gpu(self) -> None:
        """
        Copy swap statistics from GPU to CPU.
        """
        self.d_swap_acceptance.copy_to_host(self.swap_acceptance)
        self.d_swap_attempts.copy_to_host(self.swap_attempts)

    def _sync_energy_drift_stats_from_gpu(self) -> dict[str, np.ndarray]:
        """
        Copy energy-drift diagnostics from the model runtime.
        """
        return self.runtime.sync_energy_drift_stats_from_gpu()

    def _allocate_measurement_storage(
        self,
        n_meas: int,
        n_derived_meas: int,
        store_primary_histories: bool,
        observable_n_blocks: int,
    ) -> None:
        """
        Ask the model runtime to allocate measurement arrays.
        """
        self.runtime.allocate_measurement_storage(
            n_meas,
            n_derived_meas,
            store_primary_histories,
            observable_n_blocks,
        )

    def _allocate_label_storage(self, record_during_equil: bool) -> None:
        """
        Allocate storage for walker temperature-position histories.
        """
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

    def _record_positions(self, t_index: int) -> None:
        """
        Record which temperature slot each walker currently occupies.
        """
        if self.d_label_positions is None:
            return
        if self._label_record_count >= self.d_label_positions.shape[0]:
            return
        record_positions_kernel[
            self.slot_blocks,
            self.threads_per_block,
        ](
            self.d_slot_of_walker,
            self.d_label_positions,
            self._label_record_count,
        )
        self._label_record_times.append(int(t_index))
        self._label_record_count += 1

    def _record_primary_observables_to_output(self, col: int) -> None:
        """
        Ask the model runtime to record primary observables.
        """
        self.runtime.record_primary_observables(
            self.d_walker_of_slot,
            col,
        )

    def _record_derived_observables_to_output(self, col: int) -> None:
        """
        Ask the model runtime to record derived observables.
        """
        self.runtime.record_derived_observables(
            self.d_betas_by_walker,
            self.d_walker_of_slot,
            col,
        )

    def _compute_round_trip_stats(
        self,
        label_pos_arr: np.ndarray,
        record_times: list[int],
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """
        Compute simple round-trip diagnostics from walker label histories.
        """
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
            for walker in range(self.R):
                p = int(pos[walker])
                state = int(rt_state[walker])
                if p == low:
                    hit_low[walker] = True
                elif p == high:
                    hit_high[walker] = True
                if state == 0:
                    if p == low:
                        rt_state[walker] = 1
                        rt_last_t[walker] = t_index
                    elif p == high:
                        rt_state[walker] = 2
                        rt_last_t[walker] = t_index
                    continue
                if state == 1:
                    if p == high:
                        commute_counts[walker] += 1
                        rt_state[walker] = 3
                    continue
                if state == 2:
                    if p == low:
                        commute_counts[walker] += 1
                        rt_state[walker] = 4
                    continue
                if state == 3:
                    if p == low and rt_last_t[walker] >= 0:
                        rt_counts[walker] += 1
                        rt_durations.append(int(t_index - rt_last_t[walker]))
                        rt_last_t[walker] = t_index
                        rt_state[walker] = 1
                    continue
                if state == 4:
                    if p == high and rt_last_t[walker] >= 0:
                        rt_counts[walker] += 1
                        rt_durations.append(int(t_index - rt_last_t[walker]))
                        rt_last_t[walker] = t_index
                        rt_state[walker] = 2
        return (
            rt_counts,
            np.asarray(rt_durations, dtype=np.int64),
            commute_counts,
            hit_low,
            hit_high,
            hit_low & hit_high,
        )

    def run(
        self,
        *,
        record_during_equil: bool = False,
        derived_observable_stride: int = 5,
        store_primary_histories: bool = True,
        observable_n_blocks: int = 20,
    ) -> dict[str, Any]:
        """
        Run equilibration, run measurements, and return a result dictionary.
        """
        derived_observable_stride = int(derived_observable_stride)
        if derived_observable_stride <= 0:
            raise ValueError("derived_observable_stride must be positive.")
        observable_n_blocks = int(observable_n_blocks)
        if observable_n_blocks <= 0:
            raise ValueError("observable_n_blocks must be positive.")
        n_meas = self.n_measure_sweeps
        self.derived_observable_measure_sweeps = np.arange(
            0,
            n_meas,
            derived_observable_stride,
            dtype=np.int32,
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
        # -------------------------
        # Equilibration
        # -------------------------
        for sweep in range(self.n_equil_sweeps):
            self._advance_model_state()
            self._maybe_recompute_energy(sweep + 1)
            if (sweep + 1) % self.sweeps_between_swaps == 0:
                self._attempt_swaps()
                t_counter += 1
                if record_during_equil and t_counter % self.record_stride == 0:
                    self._record_positions(t_counter)

        # Optional model-specific reset after equilibration.
        reset_local_acceptance = getattr(
            self.runtime,
            "reset_local_acceptance_stats",
            None,
        )
        if reset_local_acceptance is not None:
            reset_local_acceptance()
        # -------------------------
        # Measurement
        # -------------------------
        for meas_idx in range(n_meas):
            self._advance_model_state()
            self._maybe_recompute_energy(
                self.n_equil_sweeps + meas_idx + 1
            )
            if (meas_idx + 1) % self.sweeps_between_swaps == 0:
                self._attempt_swaps()
                t_counter += 1

                if t_counter % self.record_stride == 0:
                    self._record_positions(t_counter)
            self._record_primary_observables_to_output(meas_idx)
            if meas_idx % derived_observable_stride == 0:
                self._record_derived_observables_to_output(derived_col)
                derived_col += 1

        # -------------------------
        # Copy results back
        # -------------------------
        measurements = self.runtime.copy_measurements_to_host()
        self._sync_swap_stats_from_gpu()
        energy_drift_stats = self._sync_energy_drift_stats_from_gpu()
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
            label_pos_arr,
            self._label_record_times,
        )
        return {
            **measurements,
            "derived_observable_measure_sweeps": (
                self.derived_observable_measure_sweeps
            ),
            "temps": self.temps,
            "betas": self.betas,
            "swap_acceptance": self.swap_acceptance,
            "swap_attempts": self.swap_attempts,
            **energy_drift_stats,
            "label_positions": label_pos_arr,
            "round_trip_counts": rt_counts,
            "round_trip_durations": rt_durations,
            "commute_counts": commute_counts,
            "hit_low": hit_low,
            "hit_high": hit_high,
            "hit_both_edges": hit_both_edges,
        }
