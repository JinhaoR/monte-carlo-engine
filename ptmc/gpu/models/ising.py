from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numba import cuda, float32
from numba.cuda.random import xoroshiro128p_uniform_float32

from ptmc.gpu.cuda_utils import fill_two_vectors_kernel, fill_vector_kernel
from ptmc.gpu.energy_drift import correct_energy_drift_kernel
from ptmc.gpu.layouts import pack_two_color_checkerboard
from ptmc.gpu.measurement_kernels import (
    accumulate_order_block_moments_by_slot_kernel,
    accumulate_scalar_block_moments_by_slot_kernel,
    block_count_and_size,
    record_scalar_by_slot_kernel,
)
from ptmc.gpu.interface import BaseModel, BaseRuntime

SHARED_REDUCTION_MAX_THREADS = 512


@cuda.jit
def ising_energy_init_kernel(spins, J, h, E_out):
    """
    Compute total Ising energy with forward bonds and field term.
    """
    tid = cuda.grid(1)
    R = spins.shape[0]
    L = spins.shape[2]
    half = spins.shape[3]
    sites_per_walker = 2 * L * half
    total = R * sites_per_walker
    if tid >= total:
        return

    r = tid // sites_per_walker
    rem = tid - r * sites_per_walker
    color = rem // (L * half)
    rem -= color * L * half
    i = rem // half
    k = rem - i * half

    ip = 0 if i + 1 == L else i + 1
    opp = 1 - color
    row_parity = (i + color) & 1
    kp = k + row_parity
    if kp == half:
        kp = 0

    s0 = float32(spins[r, color, i, k])
    sx = float32(spins[r, opp, ip, k])
    sy = float32(spins[r, opp, i, kp])

    cuda.atomic.add(E_out, r, -J * s0 * sx - J * s0 * sy - h * s0)


@cuda.jit
def ising_magnetization_init_kernel(spins, M_out):
    """
    Compute total Ising magnetization per walker.
    """
    tid = cuda.grid(1)
    R = spins.shape[0]
    L = spins.shape[2]
    half = spins.shape[3]
    sites_per_walker = 2 * L * half
    total = R * sites_per_walker
    if tid >= total:
        return

    r = tid // sites_per_walker
    rem = tid - r * sites_per_walker
    color = rem // (L * half)
    rem -= color * L * half
    i = rem // half
    k = rem - i * half
    cuda.atomic.add(M_out, r, float32(spins[r, color, i, k]))


@cuda.jit
def ising_update_kernel(
    spins,
    betas_by_walker,
    rng_states,
    color,
    J,
    h,
    E,
    M,
    local_update_attempts,
    local_update_acceptance,
):
    """
    One two-color checkerboard half-sweep of Ising spin flips.
    """
    tx = cuda.threadIdx.x
    r = cuda.blockIdx.y
    L = spins.shape[2]
    half = spins.shape[3]
    sites_per_color = L * half
    site = cuda.blockIdx.x * cuda.blockDim.x + tx
    if r >= spins.shape[0] or site >= sites_per_color:
        return

    i = site // half
    k = site - i * half

    ip = 0 if i + 1 == L else i + 1
    im = L - 1 if i == 0 else i - 1
    opp = 1 - color
    row_parity = (i + color) & 1
    kp = k + row_parity
    if kp == half:
        kp = 0
    km = k - (1 - row_parity)
    if km < 0:
        km = half - 1

    s0_i8 = spins[r, color, i, k]
    s0 = float32(s0_i8)
    nn_sum = float32(
        spins[r, opp, ip, k]
        + spins[r, opp, im, k]
        + spins[r, opp, i, kp]
        + spins[r, opp, i, km]
    )
    dE = float32(2.0) * s0 * (J * nn_sum + h)

    cuda.atomic.add(local_update_attempts, r, 1)
    accepted = dE <= float32(0.0)
    if not accepted:
        rng_idx = r * sites_per_color + site
        u = xoroshiro128p_uniform_float32(rng_states, rng_idx)
        accepted = u < float32(math.exp(-betas_by_walker[r] * dE))

    if accepted:
        spins[r, color, i, k] = -s0_i8
        cuda.atomic.add(E, r, dE)
        cuda.atomic.add(M, r, -float32(2.0) * s0)
        cuda.atomic.add(local_update_acceptance, r, 1)


@dataclass(frozen=True)
class IsingModel(BaseModel):
    """
    Square-lattice Ising model with two-color checkerboard updates.
    """

    J: float = 1.0
    h: float = 0.0
    ordered_start: bool = False
    name: str = "ising"
    output_prefix: str = "ising2d"

    def kernel_J(self) -> np.float32:
        return np.float32(self.J)

    def kernel_h(self) -> np.float32:
        return np.float32(self.h)

    def max_threads_per_block(self) -> int:
        return SHARED_REDUCTION_MAX_THREADS

    def validate_lattice(self, L: int) -> None:
        super().validate_lattice(L)
        if int(L) % 2 != 0:
            raise ValueError("IsingModel requires even L for two-color updates.")

    def update_sites_per_walker(self, L: int) -> int:
        self.validate_lattice(L)
        return int(L) * (int(L) // 2)

    def metadata(self) -> dict[str, Any]:
        exact_Tc = (
            2.0 * float(self.J) / np.log(1.0 + np.sqrt(2.0))
            if float(self.h) == 0.0 and float(self.J) > 0.0
            else np.nan
        )
        return {
            "model_name": self.name,
            "output_prefix": self.output_prefix,
            "J": float(self.J),
            "h": float(self.h),
            "ordered_start": bool(self.ordered_start),
            "hamiltonian": "H = -J sum_<ij> s_i s_j - h sum_i s_i",
            "update_scheme": "two-color checkerboard Metropolis spin flips",
            "exact_Tc_h0": float(exact_Tc),
            "derived_observables": ["z2_magnetization"],
            "analysis_capabilities": {
                "thermodynamics": True,
                "z2_order_parameter": True,
                "helicity_modulus": False,
            },
        }

    def create_runtime(
        self,
        *,
        L: int,
        R: int,
        rng: np.random.Generator,
        field_step: float,
        threads_per_block: int,
        full_site_blocks: int,
        update_blocks_per_walker: int,
        slot_blocks: int,
        full_lattice_blocks_per_walker: int,
        inv_N: np.float32,
    ) -> "IsingRuntime":
        del field_step
        del full_lattice_blocks_per_walker
        return IsingRuntime(
            model=self,
            L=L,
            R=R,
            rng=rng,
            threads_per_block=threads_per_block,
            full_site_blocks=full_site_blocks,
            update_blocks_per_walker=update_blocks_per_walker,
            slot_blocks=slot_blocks,
            inv_N=inv_N,
        )


class IsingRuntime(BaseRuntime):
    """
    Live CUDA state for IsingModel.
    """

    def __init__(
        self,
        *,
        model: IsingModel,
        L: int,
        R: int,
        rng: np.random.Generator,
        threads_per_block: int,
        full_site_blocks: int,
        update_blocks_per_walker: int,
        slot_blocks: int,
        inv_N: np.float32,
    ):
        self.model = model
        self.L = int(L)
        self.R = int(R)
        self.threads_per_block = int(threads_per_block)
        self.model.validate_lattice(self.L)
        if self.R <= 0:
            raise ValueError("IsingRuntime requires at least one walker.")
        if self.threads_per_block > SHARED_REDUCTION_MAX_THREADS:
            raise ValueError(
                f"threads_per_block exceeds {SHARED_REDUCTION_MAX_THREADS}."
            )
        if self.threads_per_block & (self.threads_per_block - 1):
            raise ValueError("threads_per_block must be a power of two.")

        self.full_site_blocks = int(full_site_blocks)
        self.update_blocks_per_walker = int(update_blocks_per_walker)
        self.slot_blocks = int(slot_blocks)
        self.inv_N = np.float32(inv_N)
        self.J = self.model.kernel_J()
        self.h = self.model.kernel_h()

        if self.model.ordered_start:
            spins_h = np.ones((self.R, self.L, self.L), dtype=np.int8)
        else:
            spins_h = rng.choice(
                np.array([-1, 1], dtype=np.int8),
                size=(self.R, self.L, self.L),
            ).astype(np.int8)

        self.d_spins = cuda.to_device(pack_two_color_checkerboard(spins_h))
        self.d_E = cuda.device_array(self.R, dtype=np.float32)
        self.d_E_recomputed = cuda.device_array(self.R, dtype=np.float32)
        self.d_M = cuda.device_array(self.R, dtype=np.float32)

        self.d_energy_drift_last = cuda.to_device(np.zeros(self.R, dtype=np.float32))
        self.d_energy_drift_max = cuda.to_device(np.zeros(self.R, dtype=np.float32))
        self.d_energy_recompute_checks = cuda.to_device(
            np.zeros(self.R, dtype=np.int64)
        )
        self.d_energy_recompute_corrections = cuda.to_device(
            np.zeros(self.R, dtype=np.int64)
        )
        self.d_local_update_attempts = cuda.to_device(
            np.zeros(self.R, dtype=np.int64)
        )
        self.d_local_update_acceptance = cuda.to_device(
            np.zeros(self.R, dtype=np.int64)
        )

        self.d_energies = None
        self.d_order_parameter = None
        self.d_energy_block_sums = None
        self.d_energy2_block_sums = None
        self.d_order_abs_block_sums = None
        self.d_order2_block_sums = None
        self.d_order4_block_sums = None
        self.observable_block_size = np.int32(0)
        self.energy_drift_last = np.zeros(self.R, dtype=np.float32)
        self.energy_drift_max = np.zeros(self.R, dtype=np.float32)
        self.energy_recompute_checks = np.zeros(self.R, dtype=np.int64)
        self.energy_recompute_corrections = np.zeros(self.R, dtype=np.int64)

        self._initialize_observables()

    @property
    def energy_by_walker(self):
        return self.d_E

    def _initialize_observables(self) -> None:
        fill_two_vectors_kernel[self.slot_blocks, self.threads_per_block](
            self.d_E,
            self.d_M,
            0.0,
        )
        ising_energy_init_kernel[self.full_site_blocks, self.threads_per_block](
            self.d_spins,
            self.J,
            self.h,
            self.d_E,
        )
        ising_magnetization_init_kernel[
            self.full_site_blocks,
            self.threads_per_block,
        ](
            self.d_spins,
            self.d_M,
        )

    def sweep(self, betas_by_walker, rng_states_updates, slot_of_walker) -> None:
        del slot_of_walker
        for color in (0, 1):
            ising_update_kernel[
                (self.update_blocks_per_walker, self.R),
                self.threads_per_block,
            ](
                self.d_spins,
                betas_by_walker,
                rng_states_updates,
                color,
                self.J,
                self.h,
                self.d_E,
                self.d_M,
                self.d_local_update_attempts,
                self.d_local_update_acceptance,
            )

    def reset_local_acceptance_stats(self) -> None:
        fill_two_vectors_kernel[self.slot_blocks, self.threads_per_block](
            self.d_local_update_attempts,
            self.d_local_update_acceptance,
            0,
        )

    def maybe_recompute_energy(
        self,
        sweeps_completed: int,
        recompute_stride: int,
        tolerance_per_site: np.float32,
    ) -> None:
        if recompute_stride <= 0 or sweeps_completed % recompute_stride != 0:
            return
        fill_vector_kernel[self.slot_blocks, self.threads_per_block](
            self.d_E_recomputed,
            0.0,
        )
        ising_energy_init_kernel[self.full_site_blocks, self.threads_per_block](
            self.d_spins,
            self.J,
            self.h,
            self.d_E_recomputed,
        )
        correct_energy_drift_kernel[self.slot_blocks, self.threads_per_block](
            self.d_E,
            self.d_E_recomputed,
            self.d_energy_drift_last,
            self.d_energy_drift_max,
            self.d_energy_recompute_checks,
            self.d_energy_recompute_corrections,
            tolerance_per_site,
            self.inv_N,
        )

    def allocate_measurement_storage(
        self,
        n_meas: int,
        n_derived_meas: int,
        store_primary_histories: bool,
        observable_n_blocks: int,
    ) -> None:
        del n_derived_meas
        n_meas = int(n_meas)
        if store_primary_histories:
            hist_shape = (self.R, n_meas)
            self.d_energies = cuda.device_array(hist_shape, dtype=np.float32)
            self.d_order_parameter = cuda.device_array(hist_shape, dtype=np.float32)
        else:
            self.d_energies = None
            self.d_order_parameter = None

        n_blocks, block_size = block_count_and_size(n_meas, observable_n_blocks)
        self.observable_block_size = np.int32(block_size)
        if n_blocks > 0:
            zeros = np.zeros((self.R, n_blocks), dtype=np.float32)
            self.d_energy_block_sums = cuda.to_device(zeros)
            self.d_energy2_block_sums = cuda.to_device(zeros)
            self.d_order_abs_block_sums = cuda.to_device(zeros)
            self.d_order2_block_sums = cuda.to_device(zeros)
            self.d_order4_block_sums = cuda.to_device(zeros)
        else:
            self.d_energy_block_sums = None
            self.d_energy2_block_sums = None
            self.d_order_abs_block_sums = None
            self.d_order2_block_sums = None
            self.d_order4_block_sums = None

    def record_primary_observables(self, walker_of_slot, col: int) -> None:
        if self.d_energies is not None and self.d_order_parameter is not None:
            record_scalar_by_slot_kernel[self.slot_blocks, self.threads_per_block](
                self.d_E,
                walker_of_slot,
                self.d_energies,
                col,
            )
            record_scalar_by_slot_kernel[self.slot_blocks, self.threads_per_block](
                self.d_M,
                walker_of_slot,
                self.d_order_parameter,
                col,
            )
        if self.d_energy_block_sums is not None:
            accumulate_scalar_block_moments_by_slot_kernel[
                self.slot_blocks,
                self.threads_per_block,
            ](
                self.d_E,
                walker_of_slot,
                self.d_energy_block_sums,
                self.d_energy2_block_sums,
                col,
                int(self.observable_block_size),
            )
            accumulate_order_block_moments_by_slot_kernel[
                self.slot_blocks,
                self.threads_per_block,
            ](
                self.d_M,
                walker_of_slot,
                self.d_order_abs_block_sums,
                self.d_order2_block_sums,
                self.d_order4_block_sums,
                col,
                int(self.observable_block_size),
            )

    def record_derived_observables(
        self,
        betas_by_walker,
        walker_of_slot,
        col: int,
    ) -> None:
        del betas_by_walker
        del walker_of_slot
        del col

    def copy_measurements_to_host(self) -> dict[str, np.ndarray]:
        energies = (
            self.d_energies.copy_to_host()
            if self.d_energies is not None
            else np.empty((self.R, 0), dtype=np.float32)
        )
        order_parameter = (
            self.d_order_parameter.copy_to_host()
            if self.d_order_parameter is not None
            else np.empty((self.R, 0), dtype=np.float32)
        )
        block_size = int(self.observable_block_size)
        if self.d_energy_block_sums is not None and block_size > 0:
            inv_block_size = np.float32(1.0 / float(block_size))
            energy_block_means = (
                self.d_energy_block_sums.copy_to_host() * inv_block_size
            )
            energy2_block_means = (
                self.d_energy2_block_sums.copy_to_host() * inv_block_size
            )
            order_abs_block_means = (
                self.d_order_abs_block_sums.copy_to_host() * inv_block_size
            )
            order2_block_means = (
                self.d_order2_block_sums.copy_to_host() * inv_block_size
            )
            order4_block_means = (
                self.d_order4_block_sums.copy_to_host() * inv_block_size
            )
        else:
            empty = np.empty((self.R, 0), dtype=np.float32)
            energy_block_means = empty
            energy2_block_means = empty
            order_abs_block_means = empty
            order2_block_means = empty
            order4_block_means = empty

        return {
            "energies": energies,
            "order_parameter": order_parameter,
            "energy_block_means": energy_block_means,
            "energy2_block_means": energy2_block_means,
            "order_abs_block_means": order_abs_block_means,
            "order2_block_means": order2_block_means,
            "order4_block_means": order4_block_means,
            "observable_block_size": self.observable_block_size,
            "local_update_attempts": (
                self.d_local_update_attempts.copy_to_host()
            ),
            "local_update_acceptance": (
                self.d_local_update_acceptance.copy_to_host()
            ),
        }

    def sync_energy_drift_stats_from_gpu(self) -> dict[str, np.ndarray]:
        self.d_energy_drift_last.copy_to_host(self.energy_drift_last)
        self.d_energy_drift_max.copy_to_host(self.energy_drift_max)
        self.d_energy_recompute_checks.copy_to_host(self.energy_recompute_checks)
        self.d_energy_recompute_corrections.copy_to_host(
            self.energy_recompute_corrections
        )
        return {
            "energy_drift": self.energy_drift_last,
            "energy_drift_abs_max": self.energy_drift_max,
            "energy_drift_recompute_count": self.energy_recompute_checks,
            "energy_drift_recompute_corrections": (
                self.energy_recompute_corrections
            ),
            "energy_drift_last": self.energy_drift_last,
            "energy_drift_max": self.energy_drift_max,
            "energy_recompute_checks": self.energy_recompute_checks,
            "energy_recompute_corrections": self.energy_recompute_corrections,
        }


__all__ = ["IsingModel", "IsingRuntime"]
