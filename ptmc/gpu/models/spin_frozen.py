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
from ptmc.gpu.layouts import pack_four_color_checkerboard
from ptmc.gpu.measurement_kernels import (
    accumulate_order_block_moments_by_slot_kernel,
    accumulate_scalar_block_moments_by_slot_kernel,
    block_count_and_size,
    record_scalar_by_slot_kernel,
)

TWO_PI = np.float32(2.0 * math.pi)
SHARED_REDUCTION_MAX_THREADS = 512


@cuda.jit(device=True)
def _abs2(re, im):
    return re * re + im * im


@cuda.jit(device=True)
def _color4(i, j):
    return 2 * (i & 1) + (j & 1)


@cuda.jit(device=True)
def _load_field(field, r, i, j):
    color = _color4(i, j)
    return field[r, color, i // 2, j // 2]


@cuda.jit(device=True)
def _local_potential(p_re, p_im, m_re, m_im):
    n_p = _abs2(p_re, p_im)
    n_m = _abs2(m_re, m_im)
    n = n_p + n_m
    return -float32(2.0) * n + n * n + float32(2.0) * n_p * n_m


@cuda.jit(device=True)
def _gradient_density(
    p0_re,
    p0_im,
    px_re,
    px_im,
    py_re,
    py_im,
    m0_re,
    m0_im,
    mx_re,
    mx_im,
    my_re,
    my_im,
):
    dxp_re = px_re - p0_re
    dxp_im = px_im - p0_im
    dyp_re = py_re - p0_re
    dyp_im = py_im - p0_im
    dxm_re = mx_re - m0_re
    dxm_im = mx_im - m0_im
    dym_re = my_re - m0_re
    dym_im = my_im - m0_im

    mix_re = dxp_re - dyp_im + dxm_re + dym_im
    mix_im = dxp_im + dyp_re + dxm_im - dym_re

    return (
        _abs2(dxp_re, dxp_im)
        + _abs2(dyp_re, dyp_im)
        + _abs2(dxm_re, dxm_im)
        + _abs2(dym_re, dym_im)
        + _abs2(mix_re, mix_im)
    )


@cuda.jit(device=True)
def _site_energy(p_re, p_im, m_re, m_im, r, i, j, K):
    L = p_re.shape[2] * 2
    ip = 0 if i + 1 == L else i + 1
    jp = 0 if j + 1 == L else j + 1

    p0_re = _load_field(p_re, r, i, j)
    p0_im = _load_field(p_im, r, i, j)
    m0_re = _load_field(m_re, r, i, j)
    m0_im = _load_field(m_im, r, i, j)

    px_re = _load_field(p_re, r, ip, j)
    px_im = _load_field(p_im, r, ip, j)
    py_re = _load_field(p_re, r, i, jp)
    py_im = _load_field(p_im, r, i, jp)
    mx_re = _load_field(m_re, r, ip, j)
    mx_im = _load_field(m_im, r, ip, j)
    my_re = _load_field(m_re, r, i, jp)
    my_im = _load_field(m_im, r, i, jp)

    return _local_potential(p0_re, p0_im, m0_re, m0_im) + K * _gradient_density(
        p0_re,
        p0_im,
        px_re,
        px_im,
        py_re,
        py_im,
        m0_re,
        m0_im,
        mx_re,
        mx_im,
        my_re,
        my_im,
    )


@cuda.jit(device=True)
def _bond_first_derivative(d_re, d_im, neighbor_re, neighbor_im):
    return float32(2.0) * (d_im * neighbor_re - d_re * neighbor_im)


@cuda.jit(device=True)
def _bond_second_derivative(d_re, d_im, neighbor_re, neighbor_im):
    neighbor_abs2 = _abs2(neighbor_re, neighbor_im)
    dot = d_re * neighbor_re + d_im * neighbor_im
    return float32(2.0) * (neighbor_abs2 - dot)


@cuda.jit(device=True)
def _helicity_contribution(
    p0_re,
    p0_im,
    px_re,
    px_im,
    py_re,
    py_im,
    m0_re,
    m0_im,
    mx_re,
    mx_im,
    my_re,
    my_im,
    K,
):
    dxp_re = px_re - p0_re
    dxp_im = px_im - p0_im
    dyp_re = py_re - p0_re
    dyp_im = py_im - p0_im
    dxm_re = mx_re - m0_re
    dxm_im = mx_im - m0_im
    dym_re = my_re - m0_re
    dym_im = my_im - m0_im

    mix_re = dxp_re - dyp_im + dxm_re + dym_im
    mix_im = dxp_im + dyp_re + dxm_im - dym_re

    Ix = _bond_first_derivative(dxp_re, dxp_im, px_re, px_im)
    Ix += _bond_first_derivative(dxm_re, dxm_im, mx_re, mx_im)
    sx_re = px_re + mx_re
    sx_im = px_im + mx_im
    Ix += float32(2.0) * (-mix_re * sx_im + mix_im * sx_re)

    Kx = _bond_second_derivative(dxp_re, dxp_im, px_re, px_im)
    Kx += _bond_second_derivative(dxm_re, dxm_im, mx_re, mx_im)
    Kx += float32(2.0) * (
        _abs2(sx_re, sx_im) - (mix_re * sx_re + mix_im * sx_im)
    )

    Iy = _bond_first_derivative(dyp_re, dyp_im, py_re, py_im)
    Iy += _bond_first_derivative(dym_re, dym_im, my_re, my_im)
    sy_re = my_re - py_re
    sy_im = my_im - py_im
    Iy += float32(2.0) * (mix_re * sy_re + mix_im * sy_im)

    Ky = _bond_second_derivative(dyp_re, dyp_im, py_re, py_im)
    Ky += _bond_second_derivative(dym_re, dym_im, my_re, my_im)
    Ky += float32(2.0) * (
        _abs2(sy_re, sy_im) - mix_re * sy_im + mix_im * sy_re
    )

    return K * Kx, K * Ix, K * Ky, K * Iy


@cuda.jit
def spin_frozen_energy_init_kernel(p_re, p_im, m_re, m_im, K, E_out):
    """
    Compute the dimensionless spin-frozen free energy per walker.
    """
    tid = cuda.grid(1)
    R = p_re.shape[0]
    L = p_re.shape[2] * 2
    sites_per_walker = L * L
    total = R * sites_per_walker
    if tid >= total:
        return

    r = tid // sites_per_walker
    site = tid - r * sites_per_walker
    i = site // L
    j = site - i * L
    cuda.atomic.add(E_out, r, _site_energy(p_re, p_im, m_re, m_im, r, i, j, K))


@cuda.jit
def spin_frozen_update_kernel(
    p_re,
    p_im,
    m_re,
    m_im,
    betas_by_walker,
    rng_states,
    color,
    field_step,
    K,
    E,
    local_update_attempts,
    local_update_acceptance,
):
    """
    One four-color checkerboard half-sweep of complex-field proposals.
    """
    sh_dE = cuda.shared.array(shape=SHARED_REDUCTION_MAX_THREADS, dtype=float32)

    tx = cuda.threadIdx.x
    r = cuda.blockIdx.y
    half = p_re.shape[2]
    L = half * 2
    sites_per_color = half * half
    site_idx = cuda.blockIdx.x * cuda.blockDim.x + tx
    dE_acc = float32(0.0)

    if r < p_re.shape[0] and site_idx < sites_per_color:
        color_i = color // 2
        color_j = color - 2 * color_i
        ci = site_idx // half
        cj = site_idx - ci * half
        i = 2 * ci + color_i
        j = 2 * cj + color_j
        im = L - 1 if i == 0 else i - 1
        jm = L - 1 if j == 0 else j - 1

        old_p_re = p_re[r, color, ci, cj]
        old_p_im = p_im[r, color, ci, cj]
        old_m_re = m_re[r, color, ci, cj]
        old_m_im = m_im[r, color, ci, cj]

        old_e = _site_energy(p_re, p_im, m_re, m_im, r, i, j, K)
        old_e += _site_energy(p_re, p_im, m_re, m_im, r, im, j, K)
        old_e += _site_energy(p_re, p_im, m_re, m_im, r, i, jm, K)

        rng_idx = r * sites_per_color + site_idx
        u = xoroshiro128p_uniform_float32(rng_states, rng_idx)
        new_p_re = old_p_re + field_step * (float32(2.0) * u - float32(1.0))
        u = xoroshiro128p_uniform_float32(rng_states, rng_idx)
        new_p_im = old_p_im + field_step * (float32(2.0) * u - float32(1.0))
        u = xoroshiro128p_uniform_float32(rng_states, rng_idx)
        new_m_re = old_m_re + field_step * (float32(2.0) * u - float32(1.0))
        u = xoroshiro128p_uniform_float32(rng_states, rng_idx)
        new_m_im = old_m_im + field_step * (float32(2.0) * u - float32(1.0))

        p_re[r, color, ci, cj] = new_p_re
        p_im[r, color, ci, cj] = new_p_im
        m_re[r, color, ci, cj] = new_m_re
        m_im[r, color, ci, cj] = new_m_im

        new_e = _site_energy(p_re, p_im, m_re, m_im, r, i, j, K)
        new_e += _site_energy(p_re, p_im, m_re, m_im, r, im, j, K)
        new_e += _site_energy(p_re, p_im, m_re, m_im, r, i, jm, K)
        dE = new_e - old_e

        cuda.atomic.add(local_update_attempts, r, 1)
        accepted = dE <= float32(0.0)
        if not accepted:
            acc = xoroshiro128p_uniform_float32(rng_states, rng_idx)
            accepted = acc < float32(math.exp(-(betas_by_walker[r] * dE)))

        if accepted:
            dE_acc = dE
            cuda.atomic.add(local_update_acceptance, r, 1)
        else:
            p_re[r, color, ci, cj] = old_p_re
            p_im[r, color, ci, cj] = old_p_im
            m_re[r, color, ci, cj] = old_m_re
            m_im[r, color, ci, cj] = old_m_im

    sh_dE[tx] = dE_acc
    cuda.syncthreads()

    stride = cuda.blockDim.x // 2
    while stride > 0:
        if tx < stride:
            sh_dE[tx] += sh_dE[tx + stride]
        cuda.syncthreads()
        stride //= 2

    if tx == 0 and r < p_re.shape[0] and sh_dE[0] != float32(0.0):
        cuda.atomic.add(E, r, sh_dE[0])


@cuda.jit
def spin_frozen_chirality_density_sums_kernel(
    p_re,
    p_im,
    m_re,
    m_im,
    chirality_sum,
    density_sum,
):
    """
    Accumulate sum_i chi_loc and sum_i n_i per walker.
    """
    sh_chi = cuda.shared.array(shape=SHARED_REDUCTION_MAX_THREADS, dtype=float32)
    sh_density = cuda.shared.array(
        shape=SHARED_REDUCTION_MAX_THREADS,
        dtype=float32,
    )

    tx = cuda.threadIdx.x
    r = cuda.blockIdx.y
    half = p_re.shape[2]
    area = 4 * half * half
    site = cuda.blockIdx.x * cuda.blockDim.x + tx

    local_chi = float32(0.0)
    local_density = float32(0.0)
    if r < p_re.shape[0] and site < area:
        color = site // (half * half)
        rem = site - color * half * half
        ci = rem // half
        cj = rem - ci * half

        n_p = _abs2(p_re[r, color, ci, cj], p_im[r, color, ci, cj])
        n_m = _abs2(m_re[r, color, ci, cj], m_im[r, color, ci, cj])
        n = n_p + n_m
        local_density = n
        if n > float32(1.0e-12):
            local_chi = (n_p - n_m) / n

    sh_chi[tx] = local_chi
    sh_density[tx] = local_density
    cuda.syncthreads()

    stride = cuda.blockDim.x // 2
    while stride > 0:
        if tx < stride:
            sh_chi[tx] += sh_chi[tx + stride]
            sh_density[tx] += sh_density[tx + stride]
        cuda.syncthreads()
        stride //= 2

    if tx == 0 and r < p_re.shape[0]:
        cuda.atomic.add(chirality_sum, r, sh_chi[0])
        cuda.atomic.add(density_sum, r, sh_density[0])


@cuda.jit
def spin_frozen_helicity_sums_kernel(
    p_re,
    p_im,
    m_re,
    m_im,
    K,
    sum_Kx,
    sum_Ix,
    sum_Ky,
    sum_Iy,
):
    """
    Accumulate common-phase helicity first and second derivatives.
    """
    sh_Kx = cuda.shared.array(shape=SHARED_REDUCTION_MAX_THREADS, dtype=float32)
    sh_Ix = cuda.shared.array(shape=SHARED_REDUCTION_MAX_THREADS, dtype=float32)
    sh_Ky = cuda.shared.array(shape=SHARED_REDUCTION_MAX_THREADS, dtype=float32)
    sh_Iy = cuda.shared.array(shape=SHARED_REDUCTION_MAX_THREADS, dtype=float32)

    tx = cuda.threadIdx.x
    r = cuda.blockIdx.y
    half = p_re.shape[2]
    L = half * 2
    area = L * L
    site = cuda.blockIdx.x * cuda.blockDim.x + tx

    local_Kx = float32(0.0)
    local_Ix = float32(0.0)
    local_Ky = float32(0.0)
    local_Iy = float32(0.0)

    if r < p_re.shape[0] and site < area:
        i = site // L
        j = site - i * L
        ip = 0 if i + 1 == L else i + 1
        jp = 0 if j + 1 == L else j + 1

        p0_re = _load_field(p_re, r, i, j)
        p0_im = _load_field(p_im, r, i, j)
        m0_re = _load_field(m_re, r, i, j)
        m0_im = _load_field(m_im, r, i, j)

        px_re = _load_field(p_re, r, ip, j)
        px_im = _load_field(p_im, r, ip, j)
        py_re = _load_field(p_re, r, i, jp)
        py_im = _load_field(p_im, r, i, jp)
        mx_re = _load_field(m_re, r, ip, j)
        mx_im = _load_field(m_im, r, ip, j)
        my_re = _load_field(m_re, r, i, jp)
        my_im = _load_field(m_im, r, i, jp)

        local_Kx, local_Ix, local_Ky, local_Iy = _helicity_contribution(
            p0_re,
            p0_im,
            px_re,
            px_im,
            py_re,
            py_im,
            m0_re,
            m0_im,
            mx_re,
            mx_im,
            my_re,
            my_im,
            K,
        )

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

    if tx == 0 and r < p_re.shape[0]:
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
class SpinFrozenModel(BaseModel):
    """
    Spin-frozen two-component chiral amplitude-fluctuating GL model.

    The dynamical fields are two complex scalars psi_+ and psi_-.  The
    dimensionless local potential has pure chiral minima with unit amplitude.
    """

    K: float = 1.0
    field_step: float = 0.35
    ordered_start: bool = False
    name: str = "spin_frozen"
    output_prefix: str = "spin_frozen2d"

    def kernel_K(self) -> np.float32:
        return np.float32(self.K)

    def max_threads_per_block(self) -> int:
        return SHARED_REDUCTION_MAX_THREADS

    def validate_lattice(self, L: int) -> None:
        super().validate_lattice(L)
        if int(L) % 2 != 0:
            raise ValueError(
                "SpinFrozenModel requires even L for four-color updates."
            )

    def update_sites_per_walker(self, L: int) -> int:
        self.validate_lattice(L)
        half = int(L) // 2
        return half * half

    def metadata(self) -> dict[str, Any]:
        return {
            "model_name": self.name,
            "output_prefix": self.output_prefix,
            "K": float(self.K),
            "field_step": float(self.field_step),
            "ordered_start": bool(self.ordered_start),
            "hamiltonian": (
                "F = sum_i {-2 n_i + n_i^2 + 2 n_{+,i} n_{-,i} + K G_i}, "
                "G_i = |Dx psi_+|^2 + |Dy psi_+|^2 "
                "+ |Dx psi_-|^2 + |Dy psi_-|^2 "
                "+ |(Dx+i Dy)psi_+ + (Dx-i Dy)psi_-|^2"
            ),
            "update_scheme": (
                "four-color checkerboard Metropolis proposals for the four "
                "real components of psi_+ and psi_-"
            ),
            "derived_observables": [
                "z2_chirality",
                "z2_binder_ratio",
                "total_density",
                "u1_common_phase_helicity_modulus",
                "bkt_intersection",
            ],
            "analysis_capabilities": {
                "thermodynamics": True,
                "z2_order_parameter": True,
                "helicity_modulus": True,
                "amplitude_fluctuations": True,
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
    ) -> "SpinFrozenRuntime":
        return SpinFrozenRuntime(
            model=self,
            L=L,
            R=R,
            rng=rng,
            field_step=self.field_step,
            threads_per_block=threads_per_block,
            full_site_blocks=full_site_blocks,
            update_blocks_per_walker=update_blocks_per_walker,
            slot_blocks=slot_blocks,
            full_lattice_blocks_per_walker=full_lattice_blocks_per_walker,
            inv_N=inv_N,
        )


class SpinFrozenRuntime(BaseRuntime):
    """
    Live CUDA state for SpinFrozenModel.
    """

    def __init__(
        self,
        *,
        model: SpinFrozenModel,
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
    ):
        self.model = model
        self.L = int(L)
        self.R = int(R)
        self.threads_per_block = int(threads_per_block)
        self.model.validate_lattice(self.L)
        if self.R <= 0:
            raise ValueError("SpinFrozenRuntime requires at least one walker.")
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
        self.K = self.model.kernel_K()
        self.field_step = np.float32(field_step)
        if not np.isfinite(self.K) or self.K < 0.0:
            raise ValueError("K must be finite and nonnegative.")
        if not np.isfinite(self.field_step) or self.field_step <= 0.0:
            raise ValueError("field_step must be finite and positive.")

        if self.model.ordered_start:
            p_re_h = np.ones((self.R, self.L, self.L), dtype=np.float32)
            p_im_h = np.zeros((self.R, self.L, self.L), dtype=np.float32)
            m_re_h = np.zeros((self.R, self.L, self.L), dtype=np.float32)
            m_im_h = np.zeros((self.R, self.L, self.L), dtype=np.float32)
        else:
            phases = rng.uniform(
                0.0,
                float(TWO_PI),
                size=(self.R, self.L, self.L),
            ).astype(np.float32)
            chirality = rng.choice(
                np.array([-1, 1], dtype=np.int8),
                size=(self.R, self.L, self.L),
            )
            c = np.cos(phases).astype(np.float32)
            s = np.sin(phases).astype(np.float32)
            plus_mask = chirality > 0
            p_re_h = np.where(plus_mask, c, 0.0).astype(np.float32)
            p_im_h = np.where(plus_mask, s, 0.0).astype(np.float32)
            m_re_h = np.where(plus_mask, 0.0, c).astype(np.float32)
            m_im_h = np.where(plus_mask, 0.0, s).astype(np.float32)

        self.d_p_re = cuda.to_device(pack_four_color_checkerboard(p_re_h))
        self.d_p_im = cuda.to_device(pack_four_color_checkerboard(p_im_h))
        self.d_m_re = cuda.to_device(pack_four_color_checkerboard(m_re_h))
        self.d_m_im = cuda.to_device(pack_four_color_checkerboard(m_im_h))

        self.d_E = cuda.device_array(self.R, dtype=np.float32)
        self.d_E_recomputed = cuda.device_array(self.R, dtype=np.float32)
        self.d_chirality_sum = cuda.device_array(self.R, dtype=np.float32)
        self.d_density_sum = cuda.device_array(self.R, dtype=np.float32)
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

        self.d_energies = None
        self.d_energy_block_sums = None
        self.d_energy2_block_sums = None
        self.d_chirality_order_parameter = None
        self.d_total_density = None
        self.d_helicities = None
        self.d_chirality_abs_block_sums = None
        self.d_chirality2_block_sums = None
        self.d_chirality4_block_sums = None
        self.d_density_block_sums = None
        self.d_density2_block_sums = None
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
        self._initialize_energy()

    @property
    def energy_by_walker(self):
        return self.d_E

    def _initialize_energy(self) -> None:
        fill_vector_kernel[self.slot_blocks, self.threads_per_block](
            self.d_E,
            0.0,
        )
        spin_frozen_energy_init_kernel[self.full_site_blocks, self.threads_per_block](
            self.d_p_re,
            self.d_p_im,
            self.d_m_re,
            self.d_m_im,
            self.K,
            self.d_E,
        )

    def sweep(self, betas_by_walker, rng_states_updates, slot_of_walker) -> None:
        del slot_of_walker
        for color in (0, 1, 2, 3):
            spin_frozen_update_kernel[
                (self.update_blocks_per_walker, self.R),
                self.threads_per_block,
            ](
                self.d_p_re,
                self.d_p_im,
                self.d_m_re,
                self.d_m_im,
                betas_by_walker,
                rng_states_updates,
                color,
                self.field_step,
                self.K,
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
        spin_frozen_energy_init_kernel[self.full_site_blocks, self.threads_per_block](
            self.d_p_re,
            self.d_p_im,
            self.d_m_re,
            self.d_m_im,
            self.K,
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
            self.d_chirality_order_parameter = cuda.device_array(
                (self.R, n_derived_meas),
                dtype=np.float32,
            )
            self.d_total_density = cuda.device_array(
                (self.R, n_derived_meas),
                dtype=np.float32,
            )
            self.d_helicities = cuda.device_array(
                (self.R, n_derived_meas),
                dtype=np.float32,
            )
        else:
            self.d_chirality_order_parameter = None
            self.d_total_density = None
            self.d_helicities = None

        n_d_blocks, d_block_size = block_count_and_size(
            n_derived_meas,
            observable_n_blocks,
        )
        self.derived_observable_block_size = np.int32(d_block_size)
        self.helicity_observable_block_size = np.int32(d_block_size)
        if n_d_blocks > 0:
            zeros = np.zeros((self.R, n_d_blocks), dtype=np.float32)
            self.d_chirality_abs_block_sums = cuda.to_device(zeros)
            self.d_chirality2_block_sums = cuda.to_device(zeros)
            self.d_chirality4_block_sums = cuda.to_device(zeros)
            self.d_density_block_sums = cuda.to_device(zeros)
            self.d_density2_block_sums = cuda.to_device(zeros)
            self.d_helicity_Kx_block_sums = cuda.to_device(zeros)
            self.d_helicity_Ix_block_sums = cuda.to_device(zeros)
            self.d_helicity_Ix2_block_sums = cuda.to_device(zeros)
            self.d_helicity_Ky_block_sums = cuda.to_device(zeros)
            self.d_helicity_Iy_block_sums = cuda.to_device(zeros)
            self.d_helicity_Iy2_block_sums = cuda.to_device(zeros)
        else:
            self.d_chirality_abs_block_sums = None
            self.d_chirality2_block_sums = None
            self.d_chirality4_block_sums = None
            self.d_density_block_sums = None
            self.d_density2_block_sums = None
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

    def _compute_chirality_density_sums(self) -> None:
        fill_two_vectors_kernel[self.slot_blocks, self.threads_per_block](
            self.d_chirality_sum,
            self.d_density_sum,
            0.0,
        )
        spin_frozen_chirality_density_sums_kernel[
            (self.full_lattice_blocks_per_walker, self.R),
            self.threads_per_block,
        ](
            self.d_p_re,
            self.d_p_im,
            self.d_m_re,
            self.d_m_im,
            self.d_chirality_sum,
            self.d_density_sum,
        )

    def _compute_helicity_sums(self) -> None:
        fill_four_vectors_kernel[self.slot_blocks, self.threads_per_block](
            self.d_sum_Kx,
            self.d_sum_Ix,
            self.d_sum_Ky,
            self.d_sum_Iy,
            0.0,
        )
        spin_frozen_helicity_sums_kernel[
            (self.full_lattice_blocks_per_walker, self.R),
            self.threads_per_block,
        ](
            self.d_p_re,
            self.d_p_im,
            self.d_m_re,
            self.d_m_im,
            self.K,
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
        if self.d_helicities is None:
            return
        self._compute_chirality_density_sums()
        self._compute_helicity_sums()

        record_scalar_by_slot_kernel[self.slot_blocks, self.threads_per_block](
            self.d_chirality_sum,
            walker_of_slot,
            self.d_chirality_order_parameter,
            col,
        )
        record_scalar_by_slot_kernel[self.slot_blocks, self.threads_per_block](
            self.d_density_sum,
            walker_of_slot,
            self.d_total_density,
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

        if self.d_chirality_abs_block_sums is not None:
            accumulate_order_block_moments_by_slot_kernel[
                self.slot_blocks,
                self.threads_per_block,
            ](
                self.d_chirality_sum,
                walker_of_slot,
                self.d_chirality_abs_block_sums,
                self.d_chirality2_block_sums,
                self.d_chirality4_block_sums,
                col,
                int(self.derived_observable_block_size),
            )
            accumulate_scalar_block_moments_by_slot_kernel[
                self.slot_blocks,
                self.threads_per_block,
            ](
                self.d_density_sum,
                walker_of_slot,
                self.d_density_block_sums,
                self.d_density2_block_sums,
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
        chirality_order_parameter = (
            self.d_chirality_order_parameter.copy_to_host()
            if self.d_chirality_order_parameter is not None
            else np.empty((self.R, 0), dtype=np.float32)
        )
        total_density = (
            self.d_total_density.copy_to_host()
            if self.d_total_density is not None
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
        if self.d_chirality_abs_block_sums is not None and d_block_size > 0:
            inv_d_block_size = np.float32(1.0 / float(d_block_size))
            chirality_abs_block_means = (
                self.d_chirality_abs_block_sums.copy_to_host() * inv_d_block_size
            )
            chirality2_block_means = (
                self.d_chirality2_block_sums.copy_to_host() * inv_d_block_size
            )
            chirality4_block_means = (
                self.d_chirality4_block_sums.copy_to_host() * inv_d_block_size
            )
            density_block_means = (
                self.d_density_block_sums.copy_to_host() * inv_d_block_size
            )
            density2_block_means = (
                self.d_density2_block_sums.copy_to_host() * inv_d_block_size
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
            chirality_abs_block_means = np.empty((self.R, 0), dtype=np.float32)
            chirality2_block_means = np.empty((self.R, 0), dtype=np.float32)
            chirality4_block_means = np.empty((self.R, 0), dtype=np.float32)
            density_block_means = np.empty((self.R, 0), dtype=np.float32)
            density2_block_means = np.empty((self.R, 0), dtype=np.float32)
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
            "chirality_order_parameter": chirality_order_parameter,
            "order_parameter": chirality_order_parameter,
            "total_density": total_density,
            "density": total_density,
            "helicities": helicities,
            "chirality_abs_block_means": chirality_abs_block_means,
            "chirality2_block_means": chirality2_block_means,
            "chirality4_block_means": chirality4_block_means,
            "order_abs_block_means": chirality_abs_block_means,
            "order2_block_means": chirality2_block_means,
            "order4_block_means": chirality4_block_means,
            "density_block_means": density_block_means,
            "density2_block_means": density2_block_means,
            "local_update_attempts": self.d_local_update_attempts.copy_to_host(),
            "local_update_acceptance": self.d_local_update_acceptance.copy_to_host(),
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


__all__ = ["SpinFrozenModel", "SpinFrozenRuntime"]
