from __future__ import annotations

import math
from typing import Any

import numpy as np

from ptmc.cpu.interface import BaseCPUModel, validate_cpu_model


def _block_count_and_size(n_samples: int, requested_blocks: int) -> tuple[int, int]:
    requested_blocks = max(0, int(requested_blocks))
    n_samples = int(n_samples)
    if n_samples <= 0 or requested_blocks <= 0:
        return 0, 0
    n_blocks = min(requested_blocks, n_samples // 2)
    if n_blocks < 2:
        n_blocks = 1
    return n_blocks, n_samples // n_blocks


def _block_means(samples: np.ndarray, n_blocks: int, block_size: int) -> np.ndarray:
    if n_blocks <= 0 or block_size <= 0:
        return np.empty((samples.shape[0], 0), dtype=np.float32)
    out = np.empty((samples.shape[0], n_blocks), dtype=np.float32)
    for block in range(n_blocks):
        start = block * block_size
        stop = start + block_size
        out[:, block] = np.mean(samples[:, start:stop], axis=1)
    return out


class ParallelTemperingCPU:
    """
    Simple CPU parallel tempering runner for benchmark/reference runs.

    The model supplies a NumPy state and a single-site Metropolis update. This
    intentionally does not mirror the GPU runtime/kernel interface.
    """

    def __init__(
        self,
        L: int,
        temps: np.ndarray,
        n_equil_sweeps: int,
        n_measure_sweeps: int,
        *,
        model: BaseCPUModel,
        sweeps_between_swaps: int = 1,
        record_stride: int = 10,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
        energy_recompute_stride: int = 0,
        energy_drift_tolerance_per_site: float | None = 1.0e-10,
    ):
        self.L = int(L)
        if self.L <= 0:
            raise ValueError("L must be positive.")
        self.model = validate_cpu_model(model)
        self.model.validate_lattice(self.L)

        self.temps = np.asarray(temps, dtype=np.float64)
        if self.temps.ndim != 1:
            raise ValueError("temps must be a one-dimensional array.")
        if not np.all(np.isfinite(self.temps)):
            raise ValueError("temps must contain only finite values.")
        if np.any(self.temps <= 0.0):
            raise ValueError("temps must contain positive temperatures.")
        if np.any(np.diff(self.temps) <= 0.0):
            raise ValueError("temps must be strictly increasing.")

        self.betas = 1.0 / self.temps
        self.R = int(self.temps.size)
        if self.R < 2:
            raise ValueError("At least two temperatures are required for PT.")

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
        self.energy_drift_tolerance_per_site = (
            None
            if energy_drift_tolerance_per_site is None
            else float(energy_drift_tolerance_per_site)
        )
        if (
            self.energy_drift_tolerance_per_site is not None
            and self.energy_drift_tolerance_per_site < 0.0
        ):
            raise ValueError(
                "energy_drift_tolerance_per_site must be non-negative or None."
            )

        self.rng = np.random.default_rng(seed) if rng is None else rng
        self.N = self.L * self.L
        self.sites_per_sweep = int(self.model.sweep_sites_per_walker(self.L))

        initial_states = [
            self.model.initial_state(self.L, self.rng)
            for _ in range(self.R)
        ]
        self.states = self.model.prepare_states(initial_states)
        self.energy_by_walker = np.asarray(
            [self.model.energy(state) for state in self.states],
            dtype=np.float64,
        )
        self.betas_by_walker = self.betas.copy()
        self.walker_of_slot = np.arange(self.R, dtype=np.int32)
        self.slot_of_walker = np.arange(self.R, dtype=np.int32)
        self._swap_parity = 0

        self.swap_acceptance = np.zeros(self.R - 1, dtype=np.int64)
        self.swap_attempts = np.zeros(self.R - 1, dtype=np.int64)
        self.local_update_attempts = np.zeros(self.R, dtype=np.int64)
        self.local_update_acceptance = np.zeros(self.R, dtype=np.int64)
        self.energy_drift = np.zeros(self.R, dtype=np.float64)
        self.energy_drift_abs_max = np.zeros(self.R, dtype=np.float64)
        self.energy_drift_recompute_count = np.zeros(self.R, dtype=np.int64)
        self.energy_drift_recompute_corrections = np.zeros(
            self.R,
            dtype=np.int64,
        )
        self._label_record_times: list[int] = []
        self._label_positions: list[np.ndarray] = []

    def _advance_model_state(self) -> None:
        shape = (self.R, self.sites_per_sweep)
        sites_by_walker = self.rng.integers(
            self.N,
            size=shape,
            dtype=np.int64,
        )
        self.model.sweep_walkers(
            states=self.states,
            betas_by_walker=self.betas_by_walker,
            energy_by_walker=self.energy_by_walker,
            sites_by_walker=sites_by_walker,
            accept_randoms=self.rng.random(size=shape),
            proposal_randoms=(
                self.rng.random(size=shape)
                if self.model.needs_proposal_randoms
                else None
            ),
            local_update_attempts=self.local_update_attempts,
            local_update_acceptance=self.local_update_acceptance,
        )

    def _maybe_recompute_energy(self, sweeps_completed: int) -> None:
        if self.energy_recompute_stride <= 0:
            return
        if sweeps_completed % self.energy_recompute_stride != 0:
            return
        exact = np.asarray(
            [self.model.energy(state) for state in self.states],
            dtype=np.float64,
        )
        drift = np.abs(exact - self.energy_by_walker)
        self.energy_drift = drift
        self.energy_drift_abs_max = np.maximum(self.energy_drift_abs_max, drift)
        self.energy_drift_recompute_count += 1
        if self.energy_drift_tolerance_per_site is None:
            correction_mask = np.ones(self.R, dtype=bool)
        else:
            correction_mask = drift / float(self.N) > self.energy_drift_tolerance_per_site
        self.energy_by_walker[correction_mask] = exact[correction_mask]
        self.energy_drift_recompute_corrections += correction_mask.astype(np.int64)

    def _attempt_swaps(self) -> None:
        for slot in range(self._swap_parity, self.R - 1, 2):
            wi = int(self.walker_of_slot[slot])
            wj = int(self.walker_of_slot[slot + 1])
            beta_i = float(self.betas[slot])
            beta_j = float(self.betas[slot + 1])
            energy_i = float(self.energy_by_walker[wi])
            energy_j = float(self.energy_by_walker[wj])
            delta = (beta_i - beta_j) * (energy_j - energy_i)
            self.swap_attempts[slot] += 1
            accept = delta <= 0.0 or self.rng.random() < math.exp(-delta)
            if accept:
                self.walker_of_slot[slot] = wj
                self.walker_of_slot[slot + 1] = wi
                self.slot_of_walker[wi] = slot + 1
                self.slot_of_walker[wj] = slot
                self.betas_by_walker[wi] = beta_j
                self.betas_by_walker[wj] = beta_i
                self.swap_acceptance[slot] += 1
        self._swap_parity ^= 1

    def _record_positions(self, t_index: int) -> None:
        self._label_positions.append(self.slot_of_walker.copy())
        self._label_record_times.append(int(t_index))

    def _measure_primary_by_slot(self) -> dict[str, np.ndarray]:
        energy = self.energy_by_walker[self.walker_of_slot]
        return {
            "energy": energy,
            "energy2": energy * energy,
        }

    def _measure_derived_by_slot(self) -> dict[str, np.ndarray]:
        values: dict[str, np.ndarray] = {}
        for slot in range(self.R):
            walker = int(self.walker_of_slot[slot])
            obs = self.model.measure_observables(
                self.states[walker],
                float(self.betas[slot]),
            )
            for key, value in obs.items():
                if key not in values:
                    values[key] = np.empty(self.R, dtype=np.float64)
                values[key][slot] = float(value)
        return values

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
                if state == 4 and p == high and rt_last_t[walker] >= 0:
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
        derived_observable_stride: int = 1,
        store_primary_histories: bool = True,
        observable_n_blocks: int = 20,
    ) -> dict[str, Any]:
        derived_observable_stride = int(derived_observable_stride)
        if derived_observable_stride <= 0:
            raise ValueError("derived_observable_stride must be positive.")
        observable_n_blocks = int(observable_n_blocks)
        if observable_n_blocks <= 0:
            raise ValueError("observable_n_blocks must be positive.")
        derived_observable_measure_sweeps = np.arange(
            0,
            self.n_measure_sweeps,
            derived_observable_stride,
            dtype=np.int32,
        )
        n_derived_meas = int(derived_observable_measure_sweeps.size)

        t_counter = 0
        for sweep in range(self.n_equil_sweeps):
            self._advance_model_state()
            self._maybe_recompute_energy(sweep + 1)
            if (sweep + 1) % self.sweeps_between_swaps == 0:
                self._attempt_swaps()
                t_counter += 1
                if record_during_equil and t_counter % self.record_stride == 0:
                    self._record_positions(t_counter)

        self.local_update_attempts.fill(0)
        self.local_update_acceptance.fill(0)

        primary_samples: dict[str, np.ndarray] = {}
        derived_samples: dict[str, np.ndarray] = {}
        derived_col = 0
        for meas_idx in range(self.n_measure_sweeps):
            self._advance_model_state()
            self._maybe_recompute_energy(self.n_equil_sweeps + meas_idx + 1)
            if (meas_idx + 1) % self.sweeps_between_swaps == 0:
                self._attempt_swaps()
                t_counter += 1
                if t_counter % self.record_stride == 0:
                    self._record_positions(t_counter)

            measured = self._measure_primary_by_slot()
            if not primary_samples:
                primary_samples = {
                    key: np.empty(
                        (self.R, self.n_measure_sweeps),
                        dtype=np.float64,
                    )
                    for key in measured
                }
            for key, value in measured.items():
                primary_samples[key][:, meas_idx] = value

            if meas_idx % derived_observable_stride == 0:
                measured = self._measure_derived_by_slot()
                if measured and not derived_samples:
                    derived_samples = {
                        key: np.empty(
                            (self.R, n_derived_meas),
                            dtype=np.float64,
                        )
                        for key in measured
                    }
                for key, value in measured.items():
                    derived_samples[key][:, derived_col] = value
                derived_col += 1

        n_blocks, block_size = _block_count_and_size(
            self.n_measure_sweeps,
            observable_n_blocks,
        )
        n_derived_blocks, derived_block_size = _block_count_and_size(
            n_derived_meas,
            observable_n_blocks,
        )
        result: dict[str, Any] = {
            "temps": self.temps.astype(np.float32),
            "betas": self.betas.astype(np.float32),
            "derived_observable_measure_sweeps": derived_observable_measure_sweeps,
            "energy_block_means": _block_means(
                primary_samples["energy"],
                n_blocks,
                block_size,
            ),
            "energy2_block_means": _block_means(
                primary_samples["energy2"],
                n_blocks,
                block_size,
            ),
            "observable_block_size": np.int32(block_size),
            "derived_observable_block_size": np.int32(derived_block_size),
            "swap_acceptance": self.swap_acceptance,
            "swap_attempts": self.swap_attempts,
            "local_update_attempts": self.local_update_attempts,
            "local_update_acceptance": self.local_update_acceptance,
            "energy_drift": self.energy_drift.astype(np.float32),
            "energy_drift_abs_max": self.energy_drift_abs_max.astype(np.float32),
            "energy_drift_recompute_count": self.energy_drift_recompute_count,
            "energy_drift_recompute_corrections": (
                self.energy_drift_recompute_corrections
            ),
            "energy_drift_last": self.energy_drift.astype(np.float32),
            "energy_drift_max": self.energy_drift_abs_max.astype(np.float32),
            "energy_recompute_checks": self.energy_drift_recompute_count,
            "energy_recompute_corrections": (
                self.energy_drift_recompute_corrections
            ),
        }
        for key, value in derived_samples.items():
            if key in {"order_parameter", "helicity"}:
                continue
            result[f"{key}_block_means"] = _block_means(
                value,
                n_derived_blocks,
                derived_block_size,
            )
        if "helicity_Kx" in derived_samples:
            result["helicity_observable_block_size"] = np.int32(
                derived_block_size
            )

        if store_primary_histories:
            result["energies"] = primary_samples["energy"].astype(np.float32)
            if "order_parameter" in derived_samples:
                result["order_parameter"] = derived_samples[
                    "order_parameter"
                ].astype(np.float32)
            if "helicity" in derived_samples:
                result["helicities"] = derived_samples["helicity"].astype(
                    np.float32
                )

        label_pos_arr = (
            np.asarray(self._label_positions, dtype=np.int32)
            if self._label_positions
            else np.zeros((0, self.R), dtype=np.int32)
        )
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
        result.update(
            {
                "label_positions": label_pos_arr,
                "round_trip_counts": rt_counts,
                "round_trip_durations": rt_durations,
                "commute_counts": commute_counts,
                "hit_low": hit_low,
                "hit_high": hit_high,
                "hit_both_edges": hit_both_edges,
            }
        )
        return result
