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
from ptmc.gpu.interface import BaseModel, BaseRuntime
from ptmc.gpu.layouts import pack_two_color_checkerboard
from ptmc.gpu.measurement_kernels import (
    accumulate_order_block_moments_by_slot_kernel,
    accumulate_scalar_block_moments_by_slot_kernel,
    block_count_and_size,
    record_scalar_by_slot_kernel,
)

TWO_PI = np.float32(2.0 * math.pi)
SHARED_REDUCTION_MAX_THREADS = 512


@cuda.jit(device=True)
def _cos_delta(c_i, s_i, c_j, s_j):
    return c_i * c_j + s_i * s_j


@cuda.jit(device=True)
def _sin_delta(c_i, s_i, c_j, s_j):
    return s_i * c_j - c_i * s_j


@cuda.jit(device=True)
def _chirality_product(chi_i, chi_j):
    return float32(chi_i) * float32(chi_j)


@cuda.jit(device=True)
def _bond_phase_coeff(chi_i, chi_j, J_theta):
    return J_theta * (float32(1.0) + _chirality_product(chi_i, chi_j))


@cuda.jit(device=True)
def _bond_energy(c_i, s_i, chi_i, c_j, s_j, chi_j, J_theta, J_chi):
    chi_prod = _chirality_product(chi_i, chi_j)
    coeff = J_theta * (float32(1.0) + chi_prod)
    return -coeff * _cos_delta(c_i, s_i, c_j, s_j) - J_chi * chi_prod


@cuda.jit(device=True)
def _theta_delta_energy(c_old, s_old, c_new, s_new, c_j, s_j, chi_i, chi_j, J_theta):
    coeff = _bond_phase_coeff(chi_i, chi_j, J_theta)
    return coeff * (
        _cos_delta(c_old, s_old, c_j, s_j)
        - _cos_delta(c_new, s_new, c_j, s_j)
    )


@cuda.jit(device=True)
def _chirality_flip_delta_energy(c_i, s_i, chi_i, c_j, s_j, chi_j, J_theta, J_chi):
    chi_prod = _chirality_product(chi_i, chi_j)
    return float32(2.0) * chi_prod * (
        J_theta * _cos_delta(c_i, s_i, c_j, s_j) + J_chi
    )


@cuda.jit
def london_energy_init_kernel(cos_thetas, sin_thetas, chiralities, J_theta, J_chi, E_out):
    """
    Compute the reduced chiral London Hamiltonian with forward bonds only.
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
    chi0 = chiralities[r, color, i, k]

    e_down = _bond_energy(
        c0,
        s0,
        chi0,
        cos_thetas[r, opp, ip, k],
        sin_thetas[r, opp, ip, k],
        chiralities[r, opp, ip, k],
        J_theta,
        J_chi,
    )
    e_right = _bond_energy(
        c0,
        s0,
        chi0,
        cos_thetas[r, opp, i, kp],
        sin_thetas[r, opp, i, kp],
        chiralities[r, opp, i, kp],
        J_theta,
        J_chi,
    )
    cuda.atomic.add(E_out, r, e_down + e_right)


@cuda.jit
def london_chirality_init_kernel(chiralities, M_out):
    """
    Compute the total chirality M=sum_i chi_i per walker.
    """
    tid = cuda.grid(1)
    R = chiralities.shape[0]
    L = chiralities.shape[2]
    half = chiralities.shape[3]
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
    cuda.atomic.add(M_out, r, float32(chiralities[r, color, i, k]))


@cuda.jit
def london_phase_update_kernel(
    cos_thetas,
    sin_thetas,
    chiralities,
    betas_by_walker,
    rng_states,
    color,
    theta_step,
    J_theta,
    E,
    local_update_attempts,
    local_update_acceptance,
    phase_update_attempts,
    phase_update_acceptance,
):
    """
    One two-color checkerboard half-sweep of U(1) phase proposals.
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
        chi0 = chiralities[r, color, i, k]
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
            chi0,
            chiralities[r, opp, ip, k],
            J_theta,
        )
        dE += _theta_delta_energy(
            c0,
            s0,
            c_new,
            s_new,
            cos_thetas[r, opp, im, k],
            sin_thetas[r, opp, im, k],
            chi0,
            chiralities[r, opp, im, k],
            J_theta,
        )
        dE += _theta_delta_energy(
            c0,
            s0,
            c_new,
            s_new,
            cos_thetas[r, opp, i, kp],
            sin_thetas[r, opp, i, kp],
            chi0,
            chiralities[r, opp, i, kp],
            J_theta,
        )
        dE += _theta_delta_energy(
            c0,
            s0,
            c_new,
            s_new,
            cos_thetas[r, opp, i, km],
            sin_thetas[r, opp, i, km],
            chi0,
            chiralities[r, opp, i, km],
            J_theta,
        )

        cuda.atomic.add(local_update_attempts, r, 1)
        cuda.atomic.add(phase_update_attempts, r, 1)
        accepted = dE <= float32(0.0)
        if not accepted:
            acc = xoroshiro128p_uniform_float32(rng_states, rng_idx)
            accepted = acc < float32(math.exp(-(beta * dE)))

        if accepted:
            cos_thetas[r, color, i, k] = c_new
            sin_thetas[r, color, i, k] = s_new
            dE_acc = dE
            cuda.atomic.add(local_update_acceptance, r, 1)
            cuda.atomic.add(phase_update_acceptance, r, 1)

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
def london_chirality_update_kernel(
    cos_thetas,
    sin_thetas,
    chiralities,
    betas_by_walker,
    rng_states,
    color,
    J_theta,
    J_chi,
    E,
    M,
    local_update_attempts,
    local_update_acceptance,
    chirality_update_attempts,
    chirality_update_acceptance,
):
    """
    One two-color checkerboard half-sweep of chirality flips.
    """
    sh_dE = cuda.shared.array(shape=SHARED_REDUCTION_MAX_THREADS, dtype=float32)
    sh_dM = cuda.shared.array(shape=SHARED_REDUCTION_MAX_THREADS, dtype=float32)

    tx = cuda.threadIdx.x
    r = cuda.blockIdx.y
    L = cos_thetas.shape[2]
    half = cos_thetas.shape[3]
    sites_per_color = L * half
    site_idx = cuda.blockIdx.x * cuda.blockDim.x + tx
    dE_acc = float32(0.0)
    dM_acc = float32(0.0)

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

        c0 = cos_thetas[r, color, i, k]
        s0 = sin_thetas[r, color, i, k]
        chi0 = chiralities[r, color, i, k]
        rng_idx = r * sites_per_color + site_idx

        dE = float32(0.0)
        dE += _chirality_flip_delta_energy(
            c0,
            s0,
            chi0,
            cos_thetas[r, opp, ip, k],
            sin_thetas[r, opp, ip, k],
            chiralities[r, opp, ip, k],
            J_theta,
            J_chi,
        )
        dE += _chirality_flip_delta_energy(
            c0,
            s0,
            chi0,
            cos_thetas[r, opp, im, k],
            sin_thetas[r, opp, im, k],
            chiralities[r, opp, im, k],
            J_theta,
            J_chi,
        )
        dE += _chirality_flip_delta_energy(
            c0,
            s0,
            chi0,
            cos_thetas[r, opp, i, kp],
            sin_thetas[r, opp, i, kp],
            chiralities[r, opp, i, kp],
            J_theta,
            J_chi,
        )
        dE += _chirality_flip_delta_energy(
            c0,
            s0,
            chi0,
            cos_thetas[r, opp, i, km],
            sin_thetas[r, opp, i, km],
            chiralities[r, opp, i, km],
            J_theta,
            J_chi,
        )

        cuda.atomic.add(local_update_attempts, r, 1)
        cuda.atomic.add(chirality_update_attempts, r, 1)
        accepted = dE <= float32(0.0)
        if not accepted:
            acc = xoroshiro128p_uniform_float32(rng_states, rng_idx)
            accepted = acc < float32(math.exp(-(betas_by_walker[r] * dE)))

        if accepted:
            chiralities[r, color, i, k] = -chi0
            dE_acc = dE
            dM_acc = -float32(2.0) * float32(chi0)
            cuda.atomic.add(local_update_acceptance, r, 1)
            cuda.atomic.add(chirality_update_acceptance, r, 1)

    sh_dE[tx] = dE_acc
    sh_dM[tx] = dM_acc
    cuda.syncthreads()

    stride = cuda.blockDim.x // 2
    while stride > 0:
        if tx < stride:
            sh_dE[tx] += sh_dE[tx + stride]
            sh_dM[tx] += sh_dM[tx + stride]
        cuda.syncthreads()
        stride //= 2

    if tx == 0 and r < cos_thetas.shape[0]:
        if sh_dE[0] != float32(0.0):
            cuda.atomic.add(E, r, sh_dE[0])
        if sh_dM[0] != float32(0.0):
            cuda.atomic.add(M, r, sh_dM[0])


@cuda.jit
def london_helicity_sums_kernel(
    cos_thetas,
    sin_thetas,
    chiralities,
    J_theta,
    sum_Kx,
    sum_Ix,
    sum_Ky,
    sum_Iy,
):
    """
    Accumulate common-phase helicity sums with chirality-dependent stiffness.
    """
    sh_Kx = cuda.shared.array(shape=SHARED_REDUCTION_MAX_THREADS, dtype=float32)
    sh_Ix = cuda.shared.array(shape=SHARED_REDUCTION_MAX_THREADS, dtype=float32)
    sh_Ky = cuda.shared.array(shape=SHARED_REDUCTION_MAX_THREADS, dtype=float32)
    sh_Iy = cuda.shared.array(shape=SHARED_REDUCTION_MAX_THREADS, dtype=float32)

    tx = cuda.threadIdx.x
    r = cuda.blockIdx.y
    L = cos_thetas.shape[2]
    half = cos_thetas.shape[3]
    area = 2 * L * half
    site = cuda.blockIdx.x * cuda.blockDim.x + tx

    local_Kx = float32(0.0)
    local_Ix = float32(0.0)
    local_Ky = float32(0.0)
    local_Iy = float32(0.0)

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
        chi0 = chiralities[r, color, i, k]

        coeff_x = _bond_phase_coeff(chi0, chiralities[r, opp, ip, k], J_theta)
        cx = cos_thetas[r, opp, ip, k]
        sx = sin_thetas[r, opp, ip, k]
        local_Kx = coeff_x * _cos_delta(c0, s0, cx, sx)
        local_Ix = coeff_x * _sin_delta(c0, s0, cx, sx)

        coeff_y = _bond_phase_coeff(chi0, chiralities[r, opp, i, kp], J_theta)
        cy = cos_thetas[r, opp, i, kp]
        sy = sin_thetas[r, opp, i, kp]
        local_Ky = coeff_y * _cos_delta(c0, s0, cy, sy)
        local_Iy = coeff_y * _sin_delta(c0, s0, cy, sy)

    sh_Kx[tx] = local_Kx
    sh_Ix[tx] = local_Ix
    sh_Ky[tx] = local_Ky
    sh_Iy[tx] = local_Iy
    cuda.syncthreads()

    stride = cuda.blockDim.x // 2
    while stride > 0:
        if tx < stride:
            sh_Kx[tx] += sh_Kx[tx + stride]
            sh_Ix[tx] += sh_Ix[tx + stride]
            sh_Ky[tx] += sh_Ky[tx + stride]
            sh_Iy[tx] += sh_Iy[tx + stride]
        cuda.syncthreads()
        stride //= 2

    if tx == 0 and r < cos_thetas.shape[0]:
        cuda.atomic.add(sum_Kx, r, sh_Kx[0])
        cuda.atomic.add(sum_Ix, r, sh_Ix[0])
        cuda.atomic.add(sum_Ky, r, sh_Ky[0])
        cuda.atomic.add(sum_Iy, r, sh_Iy[0])


@cuda.jit
def record_helicity_by_slot_kernel(
    sum_Kx,
    sum_Ix,
    sum_Ky,
    sum_Iy,
    betas_by_walker,
    walker_of_slot,
    inv_N,
    out,
    col,
):
    slot = cuda.grid(1)
    if slot >= walker_of_slot.shape[0]:
        return

    walker = walker_of_slot[slot]
    beta = betas_by_walker[walker]
    Kx = sum_Kx[walker]
    Ix = sum_Ix[walker]
    Ky = sum_Ky[walker]
    Iy = sum_Iy[walker]
    Yx = (Kx - beta * Ix * Ix) * inv_N
    Yy = (Ky - beta * Iy * Iy) * inv_N
    out[slot, col] = float32(0.5) * (Yx + Yy)


@cuda.jit
def accumulate_helicity_block_moments_by_slot_kernel(
    sum_Kx,
    sum_Ix,
    sum_Ky,
    sum_Iy,
    walker_of_slot,
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
    Kx = sum_Kx[walker]
    Ix = sum_Ix[walker]
    Ky = sum_Ky[walker]
    Iy = sum_Iy[walker]
    Kx_block_sums[slot, block] += Kx
    Ix_block_sums[slot, block] += Ix
    Ix2_block_sums[slot, block] += Ix * Ix
    Ky_block_sums[slot, block] += Ky
    Iy_block_sums[slot, block] += Iy
    Iy2_block_sums[slot, block] += Iy * Iy


@dataclass(frozen=True)
class LondonModel(BaseModel):
    """
    Reduced chiral London-limit U(1) x Z2 model.
    """

    J_theta: float = 1.0
    J_chi: float = 1.0
    theta_step: float = math.pi / 2.0
    ordered_start: bool = False
    name: str = "chiral_london"
    output_prefix: str = "chiral_london2d"

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.J_theta)):
            raise ValueError("J_theta must be finite.")
        if not math.isfinite(float(self.J_chi)):
            raise ValueError("J_chi must be finite.")
        if float(self.J_theta) < 0.0:
            raise ValueError("J_theta must be nonnegative.")
        if float(self.J_chi) < 0.0:
            raise ValueError("J_chi must be nonnegative.")
        if not math.isfinite(float(self.theta_step)) or float(self.theta_step) <= 0.0:
            raise ValueError("theta_step must be finite and positive.")

    def kernel_J_theta(self) -> np.float32:
        return np.float32(self.J_theta)

    def kernel_J_chi(self) -> np.float32:
        return np.float32(self.J_chi)

    def max_threads_per_block(self) -> int:
        return SHARED_REDUCTION_MAX_THREADS

    def validate_lattice(self, L: int) -> None:
        super().validate_lattice(L)
        if int(L) % 2 != 0:
            raise ValueError("LondonModel requires even L for two-color updates.")

    def update_sites_per_walker(self, L: int) -> int:
        self.validate_lattice(L)
        return int(L) * (int(L) // 2)

    def metadata(self) -> dict[str, Any]:
        return {
            "model_name": self.name,
            "output_prefix": self.output_prefix,
            "J_theta": float(self.J_theta),
            "J_chi": float(self.J_chi),
            "theta_step": float(self.theta_step),
            "ordered_start": bool(self.ordered_start),
            "hamiltonian": (
                "H = -J_theta sum_<ij> (1 + chi_i chi_j) "
                "cos(theta_i - theta_j) - J_chi sum_<ij> chi_i chi_j"
            ),
            "update_scheme": (
                "two-color checkerboard Metropolis phase proposals and "
                "chirality flips"
            ),
            "fields": "theta_i in U(1), chi_i = +/-1",
            "derived_observables": [
                "z2_chirality",
                "z2_binder_ratio",
                "u1_common_phase_helicity_modulus",
                "bkt_intersection",
            ],
            "analysis_capabilities": {
                "thermodynamics": True,
                "z2_order_parameter": True,
                "helicity_modulus": True,
                "amplitude_fluctuations": False,
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
    ) -> "LondonRuntime":
        return LondonRuntime(
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


class LondonRuntime(BaseRuntime):
    """
    Live CUDA state for LondonModel.
    """

    def __init__(
        self,
        *,
        model: LondonModel,
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
            raise ValueError("LondonRuntime requires at least one walker.")
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
        self.J_theta = self.model.kernel_J_theta()
        self.J_chi = self.model.kernel_J_chi()
        self.theta_step = np.float32(theta_step)

        if self.model.ordered_start:
            thetas_h = np.zeros((self.R, self.L, self.L), dtype=np.float32)
            chi_h = np.ones((self.R, self.L, self.L), dtype=np.int8)
        else:
            thetas_h = rng.uniform(
                0.0,
                float(TWO_PI),
                size=(self.R, self.L, self.L),
            ).astype(np.float32)
            chi_h = rng.choice(
                np.array([-1, 1], dtype=np.int8),
                size=(self.R, self.L, self.L),
            ).astype(np.int8)

        self.d_cos_thetas = cuda.to_device(
            pack_two_color_checkerboard(np.cos(thetas_h).astype(np.float32))
        )
        self.d_sin_thetas = cuda.to_device(
            pack_two_color_checkerboard(np.sin(thetas_h).astype(np.float32))
        )
        self.d_chiralities = cuda.to_device(pack_two_color_checkerboard(chi_h))

        self.d_E = cuda.device_array(self.R, dtype=np.float32)
        self.d_E_recomputed = cuda.device_array(self.R, dtype=np.float32)
        self.d_M = cuda.device_array(self.R, dtype=np.float32)
        self.d_sum_Kx = cuda.device_array(self.R, dtype=np.float32)
        self.d_sum_Ix = cuda.device_array(self.R, dtype=np.float32)
        self.d_sum_Ky = cuda.device_array(self.R, dtype=np.float32)
        self.d_sum_Iy = cuda.device_array(self.R, dtype=np.float32)

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
        self.d_phase_update_attempts = cuda.to_device(
            np.zeros(self.R, dtype=np.int64)
        )
        self.d_phase_update_acceptance = cuda.to_device(
            np.zeros(self.R, dtype=np.int64)
        )
        self.d_chirality_update_attempts = cuda.to_device(
            np.zeros(self.R, dtype=np.int64)
        )
        self.d_chirality_update_acceptance = cuda.to_device(
            np.zeros(self.R, dtype=np.int64)
        )

        self.d_energies = None
        self.d_energy_block_sums = None
        self.d_energy2_block_sums = None
        self.d_order_parameter = None
        self.d_order_abs_block_sums = None
        self.d_order2_block_sums = None
        self.d_order4_block_sums = None
        self.d_helicities = None
        self.d_helicity_Kx_block_sums = None
        self.d_helicity_Ix_block_sums = None
        self.d_helicity_Ix2_block_sums = None
        self.d_helicity_Ky_block_sums = None
        self.d_helicity_Iy_block_sums = None
        self.d_helicity_Iy2_block_sums = None

        self.observable_block_size = np.int32(0)
        self.derived_observable_block_size = np.int32(0)
        self.helicity_observable_block_size = np.int32(0)
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
        london_energy_init_kernel[self.full_site_blocks, self.threads_per_block](
            self.d_cos_thetas,
            self.d_sin_thetas,
            self.d_chiralities,
            self.J_theta,
            self.J_chi,
            self.d_E,
        )
        london_chirality_init_kernel[
            self.full_site_blocks,
            self.threads_per_block,
        ](
            self.d_chiralities,
            self.d_M,
        )

    def sweep(self, betas_by_walker, rng_states_updates, slot_of_walker) -> None:
        del slot_of_walker
        for color in (0, 1):
            london_phase_update_kernel[
                (self.update_blocks_per_walker, self.R),
                self.threads_per_block,
            ](
                self.d_cos_thetas,
                self.d_sin_thetas,
                self.d_chiralities,
                betas_by_walker,
                rng_states_updates,
                color,
                self.theta_step,
                self.J_theta,
                self.d_E,
                self.d_local_update_attempts,
                self.d_local_update_acceptance,
                self.d_phase_update_attempts,
                self.d_phase_update_acceptance,
            )
            london_chirality_update_kernel[
                (self.update_blocks_per_walker, self.R),
                self.threads_per_block,
            ](
                self.d_cos_thetas,
                self.d_sin_thetas,
                self.d_chiralities,
                betas_by_walker,
                rng_states_updates,
                color,
                self.J_theta,
                self.J_chi,
                self.d_E,
                self.d_M,
                self.d_local_update_attempts,
                self.d_local_update_acceptance,
                self.d_chirality_update_attempts,
                self.d_chirality_update_acceptance,
            )

    def reset_local_acceptance_stats(self) -> None:
        fill_two_vectors_kernel[self.slot_blocks, self.threads_per_block](
            self.d_local_update_attempts,
            self.d_local_update_acceptance,
            0,
        )
        fill_two_vectors_kernel[self.slot_blocks, self.threads_per_block](
            self.d_phase_update_attempts,
            self.d_phase_update_acceptance,
            0,
        )
        fill_two_vectors_kernel[self.slot_blocks, self.threads_per_block](
            self.d_chirality_update_attempts,
            self.d_chirality_update_acceptance,
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
        london_energy_init_kernel[self.full_site_blocks, self.threads_per_block](
            self.d_cos_thetas,
            self.d_sin_thetas,
            self.d_chiralities,
            self.J_theta,
            self.J_chi,
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

        if n_derived_meas > 0:
            self.d_order_parameter = cuda.device_array(
                (self.R, n_derived_meas),
                dtype=np.float32,
            )
            self.d_helicities = cuda.device_array(
                (self.R, n_derived_meas),
                dtype=np.float32,
            )
        else:
            self.d_order_parameter = None
            self.d_helicities = None

        n_d_blocks, d_block_size = block_count_and_size(
            n_derived_meas,
            observable_n_blocks,
        )
        self.derived_observable_block_size = np.int32(d_block_size)
        self.helicity_observable_block_size = np.int32(d_block_size)
        if n_d_blocks > 0:
            zeros = np.zeros((self.R, n_d_blocks), dtype=np.float32)
            self.d_order_abs_block_sums = cuda.to_device(zeros)
            self.d_order2_block_sums = cuda.to_device(zeros)
            self.d_order4_block_sums = cuda.to_device(zeros)
            self.d_helicity_Kx_block_sums = cuda.to_device(zeros)
            self.d_helicity_Ix_block_sums = cuda.to_device(zeros)
            self.d_helicity_Ix2_block_sums = cuda.to_device(zeros)
            self.d_helicity_Ky_block_sums = cuda.to_device(zeros)
            self.d_helicity_Iy_block_sums = cuda.to_device(zeros)
            self.d_helicity_Iy2_block_sums = cuda.to_device(zeros)
        else:
            self.d_order_abs_block_sums = None
            self.d_order2_block_sums = None
            self.d_order4_block_sums = None
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
            self.d_sum_Kx,
            self.d_sum_Ix,
            self.d_sum_Ky,
            self.d_sum_Iy,
            0.0,
        )
        london_helicity_sums_kernel[
            (self.full_lattice_blocks_per_walker, self.R),
            self.threads_per_block,
        ](
            self.d_cos_thetas,
            self.d_sin_thetas,
            self.d_chiralities,
            self.J_theta,
            self.d_sum_Kx,
            self.d_sum_Ix,
            self.d_sum_Ky,
            self.d_sum_Iy,
        )

    def record_derived_observables(
        self,
        betas_by_walker,
        walker_of_slot,
        col: int,
    ) -> None:
        if self.d_order_parameter is None or self.d_helicities is None:
            return
        self._compute_helicity_sums()

        record_scalar_by_slot_kernel[self.slot_blocks, self.threads_per_block](
            self.d_M,
            walker_of_slot,
            self.d_order_parameter,
            col,
        )
        record_helicity_by_slot_kernel[self.slot_blocks, self.threads_per_block](
            self.d_sum_Kx,
            self.d_sum_Ix,
            self.d_sum_Ky,
            self.d_sum_Iy,
            betas_by_walker,
            walker_of_slot,
            self.inv_N,
            self.d_helicities,
            col,
        )

        if self.d_order_abs_block_sums is not None:
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
                int(self.derived_observable_block_size),
            )
            accumulate_helicity_block_moments_by_slot_kernel[
                self.slot_blocks,
                self.threads_per_block,
            ](
                self.d_sum_Kx,
                self.d_sum_Ix,
                self.d_sum_Ky,
                self.d_sum_Iy,
                walker_of_slot,
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
        order_parameter = (
            self.d_order_parameter.copy_to_host()
            if self.d_order_parameter is not None
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

        d_block_size = int(self.derived_observable_block_size)
        if self.d_order_abs_block_sums is not None and d_block_size > 0:
            inv_d_block_size = np.float32(1.0 / float(d_block_size))
            order_abs_block_means = (
                self.d_order_abs_block_sums.copy_to_host() * inv_d_block_size
            )
            order2_block_means = (
                self.d_order2_block_sums.copy_to_host() * inv_d_block_size
            )
            order4_block_means = (
                self.d_order4_block_sums.copy_to_host() * inv_d_block_size
            )
            helicity_Kx_block_means = (
                self.d_helicity_Kx_block_sums.copy_to_host() * inv_d_block_size
            )
            helicity_Ix_block_means = (
                self.d_helicity_Ix_block_sums.copy_to_host() * inv_d_block_size
            )
            helicity_Ix2_block_means = (
                self.d_helicity_Ix2_block_sums.copy_to_host() * inv_d_block_size
            )
            helicity_Ky_block_means = (
                self.d_helicity_Ky_block_sums.copy_to_host() * inv_d_block_size
            )
            helicity_Iy_block_means = (
                self.d_helicity_Iy_block_sums.copy_to_host() * inv_d_block_size
            )
            helicity_Iy2_block_means = (
                self.d_helicity_Iy2_block_sums.copy_to_host() * inv_d_block_size
            )
        else:
            order_abs_block_means = np.empty((self.R, 0), dtype=np.float32)
            order2_block_means = np.empty((self.R, 0), dtype=np.float32)
            order4_block_means = np.empty((self.R, 0), dtype=np.float32)
            helicity_Kx_block_means = np.empty((self.R, 0), dtype=np.float32)
            helicity_Ix_block_means = np.empty((self.R, 0), dtype=np.float32)
            helicity_Ix2_block_means = np.empty((self.R, 0), dtype=np.float32)
            helicity_Ky_block_means = np.empty((self.R, 0), dtype=np.float32)
            helicity_Iy_block_means = np.empty((self.R, 0), dtype=np.float32)
            helicity_Iy2_block_means = np.empty((self.R, 0), dtype=np.float32)

        return {
            "energies": energies,
            "energy_block_means": energy_block_means,
            "energy2_block_means": energy2_block_means,
            "chirality_order_parameter": order_parameter,
            "z2_order_parameter": order_parameter,
            "order_parameter": order_parameter,
            "helicities": helicities,
            "chirality_abs_block_means": order_abs_block_means,
            "chirality2_block_means": order2_block_means,
            "chirality4_block_means": order4_block_means,
            "z2_abs_block_means": order_abs_block_means,
            "z2_2_block_means": order2_block_means,
            "z2_4_block_means": order4_block_means,
            "order_abs_block_means": order_abs_block_means,
            "order2_block_means": order2_block_means,
            "order4_block_means": order4_block_means,
            "local_update_attempts": self.d_local_update_attempts.copy_to_host(),
            "local_update_acceptance": self.d_local_update_acceptance.copy_to_host(),
            "phase_update_attempts": self.d_phase_update_attempts.copy_to_host(),
            "phase_update_acceptance": self.d_phase_update_acceptance.copy_to_host(),
            "chirality_update_attempts": (
                self.d_chirality_update_attempts.copy_to_host()
            ),
            "chirality_update_acceptance": (
                self.d_chirality_update_acceptance.copy_to_host()
            ),
            "helicity_Kx_block_means": helicity_Kx_block_means,
            "helicity_Ix_block_means": helicity_Ix_block_means,
            "helicity_Ix2_block_means": helicity_Ix2_block_means,
            "helicity_Ky_block_means": helicity_Ky_block_means,
            "helicity_Iy_block_means": helicity_Iy_block_means,
            "helicity_Iy2_block_means": helicity_Iy2_block_means,
            "observable_block_size": self.observable_block_size,
            "derived_observable_block_size": self.derived_observable_block_size,
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

__all__ = ["LondonModel", "LondonRuntime"]
