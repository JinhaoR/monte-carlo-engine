from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from numba import cuda, float32
from numba.cuda.random import xoroshiro128p_uniform_float32

from ptmc.gpu.cuda_utils import (
    fill_four_vectors_kernel,
    fill_two_vectors_kernel,
    fill_vector_kernel,
)
from ptmc.gpu.energy_drift import correct_energy_drift_kernel
from ptmc.gpu.layouts import pack_two_color_checkerboard
from ptmc.gpu.measurement_kernels import (
    accumulate_scalar_block_moments_by_slot_kernel,
    block_count_and_size,
    record_scalar_by_slot_kernel,
)
from ptmc.gpu.interface import BaseModel, BaseRuntime

TWO_PI = np.float32(2.0 * math.pi)
SHARED_REDUCTION_MAX_THREADS = 512


@cuda.jit(device=True)
def _cos_delta(c_i, s_i, c_j, s_j):
    return c_i * c_j + s_i * s_j


@cuda.jit(device=True)
def _sin_delta(c_i, s_i, c_j, s_j):
    return s_i * c_j - c_i * s_j


@cuda.jit(device=True)
def _bond_energy(c_i, s_i, c_j, s_j, J):
    return -J * _cos_delta(c_i, s_i, c_j, s_j)


@cuda.jit(device=True)
def _theta_delta_energy(c_old, s_old, c_new, s_new, c_j, s_j, J):
    return J * (
        _cos_delta(c_old, s_old, c_j, s_j)
        - _cos_delta(c_new, s_new, c_j, s_j)
    )


@cuda.jit
def xy_energy_init_kernel(cos_thetas, sin_thetas, J, E_out):
    """
    Compute total XY energy with forward bonds only, accumulated per walker.
    """
    tid = cuda.grid(1)
    R = cos_thetas.shape[0]
    L = cos_thetas.shape[2]
    half = cos_thetas.shape[3]
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

    c0 = cos_thetas[r, color, i, k]
    s0 = sin_thetas[r, color, i, k]

    e_down = _bond_energy(
        c0,
        s0,
        cos_thetas[r, opp, ip, k],
        sin_thetas[r, opp, ip, k],
        J,
    )
    e_right = _bond_energy(
        c0,
        s0,
        cos_thetas[r, opp, i, kp],
        sin_thetas[r, opp, i, kp],
        J,
    )
    cuda.atomic.add(E_out, r, e_down + e_right)


@cuda.jit
def xy_theta_update_kernel(
    cos_thetas,
    sin_thetas,
    betas_by_walker,
    rng_states,
    color,
    theta_step,
    J,
    E,
    local_update_attempts,
    local_update_acceptance,
):
    """
    One two-color checkerboard half-sweep of XY angle proposals.
    """
    sh_dE = cuda.shared.array(shape=SHARED_REDUCTION_MAX_THREADS, dtype=float32)

    tx = cuda.threadIdx.x
    r = cuda.blockIdx.y
    L = cos_thetas.shape[2]
    half = cos_thetas.shape[3]
    sites_per_color = L * half
    site_idx = cuda.blockIdx.x * cuda.blockDim.x + tx
    dE_acc = float32(0.0)

    if r < cos_thetas.shape[0] and site_idx < sites_per_color:
        i = site_idx // half
        k = site_idx - i * half
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

        beta = betas_by_walker[r]
        c0 = cos_thetas[r, color, i, k]
        s0 = sin_thetas[r, color, i, k]
        rng_idx = r * sites_per_color + site_idx

        u = xoroshiro128p_uniform_float32(rng_states, rng_idx)
        dtheta = theta_step * (float32(2.0) * u - float32(1.0))
        sin_d = float32(math.sin(dtheta))
        cos_d = float32(math.cos(dtheta))
        c_new = c0 * cos_d - s0 * sin_d
        s_new = s0 * cos_d + c0 * sin_d
        norm2 = c_new * c_new + s_new * s_new
        inv_norm = float32(1.0) / float32(math.sqrt(norm2))
        c_new *= inv_norm
        s_new *= inv_norm

        dE = float32(0.0)
        dE += _theta_delta_energy(
            c0,
            s0,
            c_new,
            s_new,
            cos_thetas[r, opp, ip, k],
            sin_thetas[r, opp, ip, k],
            J,
        )
        dE += _theta_delta_energy(
            c0,
            s0,
            c_new,
            s_new,
            cos_thetas[r, opp, im, k],
            sin_thetas[r, opp, im, k],
            J,
        )
        dE += _theta_delta_energy(
            c0,
            s0,
            c_new,
            s_new,
            cos_thetas[r, opp, i, kp],
            sin_thetas[r, opp, i, kp],
            J,
        )
        dE += _theta_delta_energy(
            c0,
            s0,
            c_new,
            s_new,
            cos_thetas[r, opp, i, km],
            sin_thetas[r, opp, i, km],
            J,
        )

        cuda.atomic.add(local_update_attempts, r, 1)
        accepted = dE <= float32(0.0)
        if not accepted:
            acc = xoroshiro128p_uniform_float32(rng_states, rng_idx)
            accepted = acc < float32(math.exp(-(beta * dE)))

        if accepted:
            cos_thetas[r, color, i, k] = c_new
            sin_thetas[r, color, i, k] = s_new
            dE_acc = dE
            cuda.atomic.add(local_update_acceptance, r, 1)

    sh_dE[tx] = dE_acc
    cuda.syncthreads()

    stride = cuda.blockDim.x // 2
    while stride > 0:
        if tx < stride:
            sh_dE[tx] += sh_dE[tx + stride]
        cuda.syncthreads()
        stride //= 2

    if tx == 0 and r < cos_thetas.shape[0] and sh_dE[0] != float32(0.0):
        cuda.atomic.add(E, r, sh_dE[0])


@cuda.jit
def xy_helicity_sums_kernel(
    cos_thetas,
    sin_thetas,
    sum_cos_x,
    sum_sin_x,
    sum_cos_y,
    sum_sin_y,
):
    """
    Accumulate XY helicity sums per walker with block-local reduction.
    """
    sh_cos_x = cuda.shared.array(shape=SHARED_REDUCTION_MAX_THREADS, dtype=float32)
    sh_sin_x = cuda.shared.array(shape=SHARED_REDUCTION_MAX_THREADS, dtype=float32)
    sh_cos_y = cuda.shared.array(shape=SHARED_REDUCTION_MAX_THREADS, dtype=float32)
    sh_sin_y = cuda.shared.array(shape=SHARED_REDUCTION_MAX_THREADS, dtype=float32)

    tx = cuda.threadIdx.x
    r = cuda.blockIdx.y
    L = cos_thetas.shape[2]
    half = cos_thetas.shape[3]
    area = 2 * L * half
    site = cuda.blockIdx.x * cuda.blockDim.x + tx

    local_cos_x = float32(0.0)
    local_sin_x = float32(0.0)
    local_cos_y = float32(0.0)
    local_sin_y = float32(0.0)

    if r < cos_thetas.shape[0] and site < area:
        color = site // (L * half)
        rem = site - color * L * half
        i = rem // half
        k = rem - i * half

        ip = 0 if i + 1 == L else i + 1
        opp = 1 - color
        row_parity = (i + color) & 1
        kp = k + row_parity
        if kp == half:
            kp = 0

        c0 = cos_thetas[r, color, i, k]
        s0 = sin_thetas[r, color, i, k]
        cx = cos_thetas[r, opp, ip, k]
        sx = sin_thetas[r, opp, ip, k]
        cy = cos_thetas[r, opp, i, kp]
        sy = sin_thetas[r, opp, i, kp]

        local_cos_x = _cos_delta(c0, s0, cx, sx)
        local_sin_x = _sin_delta(c0, s0, cx, sx)
        local_cos_y = _cos_delta(c0, s0, cy, sy)
        local_sin_y = _sin_delta(c0, s0, cy, sy)

    sh_cos_x[tx] = local_cos_x
    sh_sin_x[tx] = local_sin_x
    sh_cos_y[tx] = local_cos_y
    sh_sin_y[tx] = local_sin_y
    cuda.syncthreads()

    stride = cuda.blockDim.x // 2
    while stride > 0:
        if tx < stride:
            sh_cos_x[tx] += sh_cos_x[tx + stride]
            sh_sin_x[tx] += sh_sin_x[tx + stride]
            sh_cos_y[tx] += sh_cos_y[tx + stride]
            sh_sin_y[tx] += sh_sin_y[tx + stride]
        cuda.syncthreads()
        stride //= 2

    if tx == 0 and r < cos_thetas.shape[0]:
        cuda.atomic.add(sum_cos_x, r, sh_cos_x[0])
        cuda.atomic.add(sum_sin_x, r, sh_sin_x[0])
        cuda.atomic.add(sum_cos_y, r, sh_cos_y[0])
        cuda.atomic.add(sum_sin_y, r, sh_sin_y[0])


@cuda.jit
def record_helicity_by_slot_kernel(
    sum_cos_x,
    sum_sin_x,
    sum_cos_y,
    sum_sin_y,
    betas_by_walker,
    walker_of_slot,
    J,
    inv_N,
    out,
    col,
):
    slot = cuda.grid(1)
    if slot >= walker_of_slot.shape[0]:
        return

    walker = walker_of_slot[slot]
    beta = betas_by_walker[walker]
    Kx = J * sum_cos_x[walker]
    Ix = J * sum_sin_x[walker]
    Ky = J * sum_cos_y[walker]
    Iy = J * sum_sin_y[walker]
    Yx = (Kx - beta * Ix * Ix) * inv_N
    Yy = (Ky - beta * Iy * Iy) * inv_N
    out[slot, col] = float32(0.5) * (Yx + Yy)


@cuda.jit
def accumulate_helicity_block_moments_by_slot_kernel(
    sum_cos_x,
    sum_sin_x,
    sum_cos_y,
    sum_sin_y,
    walker_of_slot,
    J,
    Kx_block_sums,
    Ix_block_sums,
    Ix2_block_sums,
    Ky_block_sums,
    Iy_block_sums,
    Iy2_block_sums,
    col,
    block_size,
):
    slot = cuda.grid(1)
    if slot >= walker_of_slot.shape[0] or block_size <= 0:
        return
    block = col // block_size
    if block >= Kx_block_sums.shape[1]:
        return

    walker = walker_of_slot[slot]
    Kx = J * sum_cos_x[walker]
    Ix = J * sum_sin_x[walker]
    Ky = J * sum_cos_y[walker]
    Iy = J * sum_sin_y[walker]
    Kx_block_sums[slot, block] += Kx
    Ix_block_sums[slot, block] += Ix
    Ix2_block_sums[slot, block] += Ix * Ix
    Ky_block_sums[slot, block] += Ky
    Iy_block_sums[slot, block] += Iy
    Iy2_block_sums[slot, block] += Iy * Iy


@dataclass(frozen=True)
class XYModel(BaseModel):
    """
    Square-lattice classical XY model with two-color checkerboard updates.
    """

    J: float = 1.0
    theta_step: float = math.pi / 2.0
    ordered_start: bool = False
    name: str = "xy"
    output_prefix: str = "xy2d"

    def kernel_J(self) -> np.float32:
        return np.float32(self.J)

    def max_threads_per_block(self) -> int:
        return SHARED_REDUCTION_MAX_THREADS

    def validate_lattice(self, L: int) -> None:
        super().validate_lattice(L)
        if int(L) % 2 != 0:
            raise ValueError("XYModel requires even L for two-color updates.")

    def update_sites_per_walker(self, L: int) -> int:
        self.validate_lattice(L)
        return int(L) * (int(L) // 2)

    def metadata(self) -> dict[str, Any]:
        return {
            "model_name": self.name,
            "output_prefix": self.output_prefix,
            "J": float(self.J),
            "theta_step": float(self.theta_step),
            "ordered_start": bool(self.ordered_start),
            "hamiltonian": "H = -J sum_<ij> cos(theta_i - theta_j)",
            "update_scheme": "two-color checkerboard Metropolis angle proposals",
            "derived_observables": ["u1_helicity_modulus", "bkt_intersection"],
            "analysis_capabilities": {
                "thermodynamics": True,
                "z2_order_parameter": False,
                "helicity_modulus": True,
            },
        }

    def create_runtime(
        self,
        *,
        L: int,
        R: int,
        rng: np.random.Generator,
        threads_per_block: int,
        full_site_blocks: int,
        update_blocks_per_walker: int,
        slot_blocks: int,
        full_lattice_blocks_per_walker: int,
        inv_N: np.float32,
    ) -> "XYRuntime":
        return XYRuntime(
            model=self,
            L=L,
            R=R,
            rng=rng,
            theta_step=self.theta_step,
            threads_per_block=threads_per_block,
            full_site_blocks=full_site_blocks,
            update_blocks_per_walker=update_blocks_per_walker,
            slot_blocks=slot_blocks,
            full_lattice_blocks_per_walker=full_lattice_blocks_per_walker,
            inv_N=inv_N,
        )


class XYRuntime(BaseRuntime):
    """
    Live CUDA state for XYModel.
    """

    def __init__(
        self,
        *,
        model: XYModel,
        L: int,
        R: int,
        rng: np.random.Generator,
        theta_step: float,
        threads_per_block: int,
        full_site_blocks: int,
        update_blocks_per_walker: int,
        slot_blocks: int,
        full_lattice_blocks_per_walker: int,
        inv_N: np.float32,
    ):
        self.model = model
        self.L = int(L)
        self.R = int(R)
        self.threads_per_block = int(threads_per_block)
        self.model.validate_lattice(self.L)
        if self.R <= 0:
            raise ValueError("XYRuntime requires at least one walker.")
        if self.threads_per_block > SHARED_REDUCTION_MAX_THREADS:
            raise ValueError(
                f"threads_per_block exceeds {SHARED_REDUCTION_MAX_THREADS}."
            )
        if self.threads_per_block & (self.threads_per_block - 1):
            raise ValueError("threads_per_block must be a power of two.")

        self.full_site_blocks = int(full_site_blocks)
        self.update_blocks_per_walker = int(update_blocks_per_walker)
        self.slot_blocks = int(slot_blocks)
        self.full_lattice_blocks_per_walker = int(full_lattice_blocks_per_walker)
        self.inv_N = np.float32(inv_N)
        self.J = self.model.kernel_J()
        self.theta_step = np.float32(theta_step)
        if not np.isfinite(self.theta_step) or self.theta_step <= 0.0:
            raise ValueError("theta_step must be finite and positive.")

        if self.model.ordered_start:
            thetas_h = np.zeros((self.R, self.L, self.L), dtype=np.float32)
        else:
            thetas_h = rng.uniform(
                0.0,
                float(TWO_PI),
                size=(self.R, self.L, self.L),
            ).astype(np.float32)

        self.d_cos_thetas = cuda.to_device(
            pack_two_color_checkerboard(np.cos(thetas_h).astype(np.float32))
        )
        self.d_sin_thetas = cuda.to_device(
            pack_two_color_checkerboard(np.sin(thetas_h).astype(np.float32))
        )

        self.d_E = cuda.device_array(self.R, dtype=np.float32)
        self.d_E_recomputed = cuda.device_array(self.R, dtype=np.float32)
        self.d_sum_cos_x = cuda.device_array(self.R, dtype=np.float32)
        self.d_sum_sin_x = cuda.device_array(self.R, dtype=np.float32)
        self.d_sum_cos_y = cuda.device_array(self.R, dtype=np.float32)
        self.d_sum_sin_y = cuda.device_array(self.R, dtype=np.float32)

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
        self.d_energy_block_sums = None
        self.d_energy2_block_sums = None
        self.d_helicities = None
        self.d_helicity_Kx_block_sums = None
        self.d_helicity_Ix_block_sums = None
        self.d_helicity_Ix2_block_sums = None
        self.d_helicity_Ky_block_sums = None
        self.d_helicity_Iy_block_sums = None
        self.d_helicity_Iy2_block_sums = None

        self.observable_block_size = np.int32(0)
        self.helicity_observable_block_size = np.int32(0)
        self.energy_drift_last = np.zeros(self.R, dtype=np.float32)
        self.energy_drift_max = np.zeros(self.R, dtype=np.float32)
        self.energy_recompute_checks = np.zeros(self.R, dtype=np.int64)
        self.energy_recompute_corrections = np.zeros(self.R, dtype=np.int64)
        self._initialize_energy()

    @property
    def energy_by_walker(self):
        return self.d_E

    def _initialize_energy(self) -> None:
        fill_vector_kernel[self.slot_blocks, self.threads_per_block](
            self.d_E,
            0.0,
        )
        xy_energy_init_kernel[self.full_site_blocks, self.threads_per_block](
            self.d_cos_thetas,
            self.d_sin_thetas,
            self.J,
            self.d_E,
        )

    def sweep(self, betas_by_walker, rng_states_updates, slot_of_walker) -> None:
        del slot_of_walker
        for color in (0, 1):
            xy_theta_update_kernel[
                (self.update_blocks_per_walker, self.R),
                self.threads_per_block,
            ](
                self.d_cos_thetas,
                self.d_sin_thetas,
                betas_by_walker,
                rng_states_updates,
                color,
                self.theta_step,
                self.J,
                self.d_E,
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
        xy_energy_init_kernel[self.full_site_blocks, self.threads_per_block](
            self.d_cos_thetas,
            self.d_sin_thetas,
            self.J,
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
        n_meas = int(n_meas)
        n_derived_meas = int(n_derived_meas)
        if store_primary_histories:
            self.d_energies = cuda.device_array((self.R, n_meas), dtype=np.float32)
        else:
            self.d_energies = None

        n_blocks, block_size = block_count_and_size(n_meas, observable_n_blocks)
        self.observable_block_size = np.int32(block_size)
        if n_blocks > 0:
            zeros = np.zeros((self.R, n_blocks), dtype=np.float32)
            self.d_energy_block_sums = cuda.to_device(zeros)
            self.d_energy2_block_sums = cuda.to_device(zeros)
        else:
            self.d_energy_block_sums = None
            self.d_energy2_block_sums = None

        self.d_helicities = (
            cuda.device_array((self.R, n_derived_meas), dtype=np.float32)
            if n_derived_meas > 0
            else None
        )

        n_h_blocks, h_block_size = block_count_and_size(
            n_derived_meas,
            observable_n_blocks,
        )
        self.helicity_observable_block_size = np.int32(h_block_size)
        if n_h_blocks > 0:
            zeros = np.zeros((self.R, n_h_blocks), dtype=np.float32)
            self.d_helicity_Kx_block_sums = cuda.to_device(zeros)
            self.d_helicity_Ix_block_sums = cuda.to_device(zeros)
            self.d_helicity_Ix2_block_sums = cuda.to_device(zeros)
            self.d_helicity_Ky_block_sums = cuda.to_device(zeros)
            self.d_helicity_Iy_block_sums = cuda.to_device(zeros)
            self.d_helicity_Iy2_block_sums = cuda.to_device(zeros)
        else:
            self.d_helicity_Kx_block_sums = None
            self.d_helicity_Ix_block_sums = None
            self.d_helicity_Ix2_block_sums = None
            self.d_helicity_Ky_block_sums = None
            self.d_helicity_Iy_block_sums = None
            self.d_helicity_Iy2_block_sums = None

    def record_primary_observables(self, walker_of_slot, col: int) -> None:
        if self.d_energies is not None:
            record_scalar_by_slot_kernel[self.slot_blocks, self.threads_per_block](
                self.d_E,
                walker_of_slot,
                self.d_energies,
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

    def _compute_helicity_sums(self) -> None:
        fill_four_vectors_kernel[self.slot_blocks, self.threads_per_block](
            self.d_sum_cos_x,
            self.d_sum_sin_x,
            self.d_sum_cos_y,
            self.d_sum_sin_y,
            0.0,
        )
        xy_helicity_sums_kernel[
            (self.full_lattice_blocks_per_walker, self.R),
            self.threads_per_block,
        ](
            self.d_cos_thetas,
            self.d_sin_thetas,
            self.d_sum_cos_x,
            self.d_sum_sin_x,
            self.d_sum_cos_y,
            self.d_sum_sin_y,
        )

    def record_derived_observables(
        self,
        betas_by_walker,
        walker_of_slot,
        col: int,
    ) -> None:
        if self.d_helicities is None:
            return
        self._compute_helicity_sums()
        record_helicity_by_slot_kernel[self.slot_blocks, self.threads_per_block](
            self.d_sum_cos_x,
            self.d_sum_sin_x,
            self.d_sum_cos_y,
            self.d_sum_sin_y,
            betas_by_walker,
            walker_of_slot,
            self.J,
            self.inv_N,
            self.d_helicities,
            col,
        )
        if self.d_helicity_Kx_block_sums is not None:
            accumulate_helicity_block_moments_by_slot_kernel[
                self.slot_blocks,
                self.threads_per_block,
            ](
                self.d_sum_cos_x,
                self.d_sum_sin_x,
                self.d_sum_cos_y,
                self.d_sum_sin_y,
                walker_of_slot,
                self.J,
                self.d_helicity_Kx_block_sums,
                self.d_helicity_Ix_block_sums,
                self.d_helicity_Ix2_block_sums,
                self.d_helicity_Ky_block_sums,
                self.d_helicity_Iy_block_sums,
                self.d_helicity_Iy2_block_sums,
                col,
                int(self.helicity_observable_block_size),
            )

    def copy_measurements_to_host(self) -> dict[str, np.ndarray]:
        energies = (
            self.d_energies.copy_to_host()
            if self.d_energies is not None
            else np.empty((self.R, 0), dtype=np.float32)
        )
        helicities = (
            self.d_helicities.copy_to_host()
            if self.d_helicities is not None
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
        else:
            energy_block_means = np.empty((self.R, 0), dtype=np.float32)
            energy2_block_means = np.empty((self.R, 0), dtype=np.float32)

        h_block_size = int(self.helicity_observable_block_size)
        if self.d_helicity_Kx_block_sums is not None and h_block_size > 0:
            inv_h_block_size = np.float32(1.0 / float(h_block_size))
            helicity_Kx_block_means = (
                self.d_helicity_Kx_block_sums.copy_to_host() * inv_h_block_size
            )
            helicity_Ix_block_means = (
                self.d_helicity_Ix_block_sums.copy_to_host() * inv_h_block_size
            )
            helicity_Ix2_block_means = (
                self.d_helicity_Ix2_block_sums.copy_to_host() * inv_h_block_size
            )
            helicity_Ky_block_means = (
                self.d_helicity_Ky_block_sums.copy_to_host() * inv_h_block_size
            )
            helicity_Iy_block_means = (
                self.d_helicity_Iy_block_sums.copy_to_host() * inv_h_block_size
            )
            helicity_Iy2_block_means = (
                self.d_helicity_Iy2_block_sums.copy_to_host() * inv_h_block_size
            )
        else:
            empty = np.empty((self.R, 0), dtype=np.float32)
            helicity_Kx_block_means = empty
            helicity_Ix_block_means = empty
            helicity_Ix2_block_means = empty
            helicity_Ky_block_means = empty
            helicity_Iy_block_means = empty
            helicity_Iy2_block_means = empty

        return {
            "energies": energies,
            "energy_block_means": energy_block_means,
            "energy2_block_means": energy2_block_means,
            "helicities": helicities,
            "local_update_attempts": (
                self.d_local_update_attempts.copy_to_host()
            ),
            "local_update_acceptance": (
                self.d_local_update_acceptance.copy_to_host()
            ),
            "helicity_Kx_block_means": helicity_Kx_block_means,
            "helicity_Ix_block_means": helicity_Ix_block_means,
            "helicity_Ix2_block_means": helicity_Ix2_block_means,
            "helicity_Ky_block_means": helicity_Ky_block_means,
            "helicity_Iy_block_means": helicity_Iy_block_means,
            "helicity_Iy2_block_means": helicity_Iy2_block_means,
            "observable_block_size": self.observable_block_size,
            "derived_observable_block_size": self.helicity_observable_block_size,
            "helicity_observable_block_size": self.helicity_observable_block_size,
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


__all__ = ["XYModel", "XYRuntime"]
