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
    accumulate_order_block_moments_by_slot_kernel,
    accumulate_scalar_block_moments_by_slot_kernel,
    block_count_and_size,
    record_scalar_by_slot_kernel,
)
from ptmc.gpu.interface import BaseModel, BaseRuntime

TWO_PI = np.float32(2.0 * math.pi)
HALF_PI = np.float32(0.5 * math.pi)
SHARED_REDUCTION_MAX_THREADS = 512


@cuda.jit(device=True)
def _cos_delta(c_i, s_i, c_j, s_j):
    return c_i * c_j + s_i * s_j


@cuda.jit(device=True)
def _sin_delta(c_i, s_i, c_j, s_j):
    return s_i * c_j - c_i * s_j


@cuda.jit(device=True)
def _cos2_relative(c1, s1, c2, s2):
    c12 = _cos_delta(c1, s1, c2, s2)
    s12 = _sin_delta(c1, s1, c2, s2)
    return c12 * c12 - s12 * s12


@cuda.jit(device=True)
def _amp1(x):
    if x <= float32(0.0):
        return float32(0.0)
    return float32(math.sqrt(x))


@cuda.jit(device=True)
def _amp2(x):
    one_minus_x = float32(1.0) - x
    if one_minus_x <= float32(0.0):
        return float32(0.0)
    return float32(math.sqrt(one_minus_x))


@cuda.jit(device=True)
def _component_bond_energy(a_i, c_i, s_i, a_j, c_j, s_j):
    return -a_i * a_j * _cos_delta(c_i, s_i, c_j, s_j)


@cuda.jit(device=True)
def _onsite_energy(x, c1, s1, c2, s2, K):
    return K * x * (float32(1.0) - x) * (
        _cos2_relative(c1, s1, c2, s2) - float32(1.0)
    )


@cuda.jit(device=True)
def _phase_delta_energy_for_neighbor(
    a_i,
    c_old,
    s_old,
    c_new,
    s_new,
    a_j,
    c_j,
    s_j,
):
    return a_i * a_j * (
        _cos_delta(c_old, s_old, c_j, s_j)
        - _cos_delta(c_new, s_new, c_j, s_j)
    )


@cuda.jit(device=True)
def _amplitude_delta_energy_for_neighbor(
    a1_old,
    a2_old,
    a1_new,
    a2_new,
    c1_i,
    s1_i,
    c2_i,
    s2_i,
    x_j,
    c1_j,
    s1_j,
    c2_j,
    s2_j,
):
    a1_j = _amp1(x_j)
    a2_j = _amp2(x_j)
    old_e = _component_bond_energy(a1_old, c1_i, s1_i, a1_j, c1_j, s1_j)
    old_e += _component_bond_energy(a2_old, c2_i, s2_i, a2_j, c2_j, s2_j)
    new_e = _component_bond_energy(a1_new, c1_i, s1_i, a1_j, c1_j, s1_j)
    new_e += _component_bond_energy(a2_new, c2_i, s2_i, a2_j, c2_j, s2_j)
    return new_e - old_e


@cuda.jit
def tbg_energy_init_kernel(c1, s1, c2, s2, amp1_sq, K, E_out):
    """
    Compute total TBG Hamiltonian using forward bonds and onsite anisotropy.
    """
    tid = cuda.grid(1)
    R = c1.shape[0]
    L = c1.shape[2]
    half = c1.shape[3]
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

    x0 = amp1_sq[r, color, i, k]
    a10 = _amp1(x0)
    a20 = _amp2(x0)
    c10 = c1[r, color, i, k]
    s10 = s1[r, color, i, k]
    c20 = c2[r, color, i, k]
    s20 = s2[r, color, i, k]

    xd = amp1_sq[r, opp, ip, k]
    xr = amp1_sq[r, opp, i, kp]

    e = float32(0.0)
    e += _component_bond_energy(
        a10,
        c10,
        s10,
        _amp1(xd),
        c1[r, opp, ip, k],
        s1[r, opp, ip, k],
    )
    e += _component_bond_energy(
        a20,
        c20,
        s20,
        _amp2(xd),
        c2[r, opp, ip, k],
        s2[r, opp, ip, k],
    )
    e += _component_bond_energy(
        a10,
        c10,
        s10,
        _amp1(xr),
        c1[r, opp, i, kp],
        s1[r, opp, i, kp],
    )
    e += _component_bond_energy(
        a20,
        c20,
        s20,
        _amp2(xr),
        c2[r, opp, i, kp],
        s2[r, opp, i, kp],
    )
    e += _onsite_energy(x0, c10, s10, c20, s20, K)
    cuda.atomic.add(E_out, r, e)


@cuda.jit
def tbg_update_kernel(
    c1,
    s1,
    c2,
    s2,
    amp1_sq,
    betas_by_walker,
    rng_states,
    color,
    phase_step,
    amplitude_step,
    K,
    E,
    local_update_attempts,
    local_update_acceptance,
    phase_update_attempts,
    phase_update_acceptance,
    amplitude_update_attempts,
    amplitude_update_acceptance,
):
    """
    One two-color checkerboard half-sweep.

    Each active site receives three local Metropolis proposals:
    phi_1, phi_2, and x = |Delta_1|^2 with |Delta_2|^2 = 1 - x.
    """
    sh_dE = cuda.shared.array(shape=SHARED_REDUCTION_MAX_THREADS, dtype=float32)

    tx = cuda.threadIdx.x
    r = cuda.blockIdx.y
    L = c1.shape[2]
    half = c1.shape[3]
    sites_per_color = L * half
    site_idx = cuda.blockIdx.x * cuda.blockDim.x + tx
    dE_acc = float32(0.0)

    if r < c1.shape[0] and site_idx < sites_per_color:
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
        rng_idx = r * sites_per_color + site_idx

        x0 = amp1_sq[r, color, i, k]
        a10 = _amp1(x0)
        a20 = _amp2(x0)
        c10 = c1[r, color, i, k]
        s10 = s1[r, color, i, k]
        c20 = c2[r, color, i, k]
        s20 = s2[r, color, i, k]

        # Component-1 phase proposal.
        u = xoroshiro128p_uniform_float32(rng_states, rng_idx)
        dtheta = phase_step * (float32(2.0) * u - float32(1.0))
        sin_d = float32(math.sin(dtheta))
        cos_d = float32(math.cos(dtheta))
        c1_new = c10 * cos_d - s10 * sin_d
        s1_new = s10 * cos_d + c10 * sin_d
        norm2 = c1_new * c1_new + s1_new * s1_new
        inv_norm = float32(1.0) / float32(math.sqrt(norm2))
        c1_new *= inv_norm
        s1_new *= inv_norm

        dE = float32(0.0)
        dE += _phase_delta_energy_for_neighbor(
            a10,
            c10,
            s10,
            c1_new,
            s1_new,
            _amp1(amp1_sq[r, opp, ip, k]),
            c1[r, opp, ip, k],
            s1[r, opp, ip, k],
        )
        dE += _phase_delta_energy_for_neighbor(
            a10,
            c10,
            s10,
            c1_new,
            s1_new,
            _amp1(amp1_sq[r, opp, im, k]),
            c1[r, opp, im, k],
            s1[r, opp, im, k],
        )
        dE += _phase_delta_energy_for_neighbor(
            a10,
            c10,
            s10,
            c1_new,
            s1_new,
            _amp1(amp1_sq[r, opp, i, kp]),
            c1[r, opp, i, kp],
            s1[r, opp, i, kp],
        )
        dE += _phase_delta_energy_for_neighbor(
            a10,
            c10,
            s10,
            c1_new,
            s1_new,
            _amp1(amp1_sq[r, opp, i, km]),
            c1[r, opp, i, km],
            s1[r, opp, i, km],
        )
        dE += _onsite_energy(x0, c1_new, s1_new, c20, s20, K)
        dE -= _onsite_energy(x0, c10, s10, c20, s20, K)

        cuda.atomic.add(local_update_attempts, r, 1)
        cuda.atomic.add(phase_update_attempts, r, 1)
        accepted = dE <= float32(0.0)
        if not accepted:
            acc = xoroshiro128p_uniform_float32(rng_states, rng_idx)
            accepted = acc < float32(math.exp(-(beta * dE)))
        if accepted:
            c1[r, color, i, k] = c1_new
            s1[r, color, i, k] = s1_new
            c10 = c1_new
            s10 = s1_new
            dE_acc += dE
            cuda.atomic.add(local_update_acceptance, r, 1)
            cuda.atomic.add(phase_update_acceptance, r, 1)

        # Component-2 phase proposal.
        u = xoroshiro128p_uniform_float32(rng_states, rng_idx)
        dtheta = phase_step * (float32(2.0) * u - float32(1.0))
        sin_d = float32(math.sin(dtheta))
        cos_d = float32(math.cos(dtheta))
        c2_new = c20 * cos_d - s20 * sin_d
        s2_new = s20 * cos_d + c20 * sin_d
        norm2 = c2_new * c2_new + s2_new * s2_new
        inv_norm = float32(1.0) / float32(math.sqrt(norm2))
        c2_new *= inv_norm
        s2_new *= inv_norm

        dE = float32(0.0)
        dE += _phase_delta_energy_for_neighbor(
            a20,
            c20,
            s20,
            c2_new,
            s2_new,
            _amp2(amp1_sq[r, opp, ip, k]),
            c2[r, opp, ip, k],
            s2[r, opp, ip, k],
        )
        dE += _phase_delta_energy_for_neighbor(
            a20,
            c20,
            s20,
            c2_new,
            s2_new,
            _amp2(amp1_sq[r, opp, im, k]),
            c2[r, opp, im, k],
            s2[r, opp, im, k],
        )
        dE += _phase_delta_energy_for_neighbor(
            a20,
            c20,
            s20,
            c2_new,
            s2_new,
            _amp2(amp1_sq[r, opp, i, kp]),
            c2[r, opp, i, kp],
            s2[r, opp, i, kp],
        )
        dE += _phase_delta_energy_for_neighbor(
            a20,
            c20,
            s20,
            c2_new,
            s2_new,
            _amp2(amp1_sq[r, opp, i, km]),
            c2[r, opp, i, km],
            s2[r, opp, i, km],
        )
        dE += _onsite_energy(x0, c10, s10, c2_new, s2_new, K)
        dE -= _onsite_energy(x0, c10, s10, c20, s20, K)

        cuda.atomic.add(local_update_attempts, r, 1)
        cuda.atomic.add(phase_update_attempts, r, 1)
        accepted = dE <= float32(0.0)
        if not accepted:
            acc = xoroshiro128p_uniform_float32(rng_states, rng_idx)
            accepted = acc < float32(math.exp(-(beta * dE)))
        if accepted:
            c2[r, color, i, k] = c2_new
            s2[r, color, i, k] = s2_new
            c20 = c2_new
            s20 = s2_new
            dE_acc += dE
            cuda.atomic.add(local_update_acceptance, r, 1)
            cuda.atomic.add(phase_update_acceptance, r, 1)

        # Relative-density/amplitude proposal, x = |Delta_1|^2.
        u = xoroshiro128p_uniform_float32(rng_states, rng_idx)
        x_new = x0 + amplitude_step * (float32(2.0) * u - float32(1.0))
        cuda.atomic.add(local_update_attempts, r, 1)
        cuda.atomic.add(amplitude_update_attempts, r, 1)

        if x_new >= float32(0.0) and x_new <= float32(1.0):
            a1_new = _amp1(x_new)
            a2_new = _amp2(x_new)
            dE = float32(0.0)
            dE += _amplitude_delta_energy_for_neighbor(
                a10,
                a20,
                a1_new,
                a2_new,
                c10,
                s10,
                c20,
                s20,
                amp1_sq[r, opp, ip, k],
                c1[r, opp, ip, k],
                s1[r, opp, ip, k],
                c2[r, opp, ip, k],
                s2[r, opp, ip, k],
            )
            dE += _amplitude_delta_energy_for_neighbor(
                a10,
                a20,
                a1_new,
                a2_new,
                c10,
                s10,
                c20,
                s20,
                amp1_sq[r, opp, im, k],
                c1[r, opp, im, k],
                s1[r, opp, im, k],
                c2[r, opp, im, k],
                s2[r, opp, im, k],
            )
            dE += _amplitude_delta_energy_for_neighbor(
                a10,
                a20,
                a1_new,
                a2_new,
                c10,
                s10,
                c20,
                s20,
                amp1_sq[r, opp, i, kp],
                c1[r, opp, i, kp],
                s1[r, opp, i, kp],
                c2[r, opp, i, kp],
                s2[r, opp, i, kp],
            )
            dE += _amplitude_delta_energy_for_neighbor(
                a10,
                a20,
                a1_new,
                a2_new,
                c10,
                s10,
                c20,
                s20,
                amp1_sq[r, opp, i, km],
                c1[r, opp, i, km],
                s1[r, opp, i, km],
                c2[r, opp, i, km],
                s2[r, opp, i, km],
            )
            dE += _onsite_energy(x_new, c10, s10, c20, s20, K)
            dE -= _onsite_energy(x0, c10, s10, c20, s20, K)

            accepted = dE <= float32(0.0)
            if not accepted:
                acc = xoroshiro128p_uniform_float32(rng_states, rng_idx)
                accepted = acc < float32(math.exp(-(beta * dE)))
            if accepted:
                amp1_sq[r, color, i, k] = x_new
                dE_acc += dE
                cuda.atomic.add(local_update_acceptance, r, 1)
                cuda.atomic.add(amplitude_update_acceptance, r, 1)

    sh_dE[tx] = dE_acc
    cuda.syncthreads()

    stride = cuda.blockDim.x // 2
    while stride > 0:
        if tx < stride:
            sh_dE[tx] += sh_dE[tx + stride]
        cuda.syncthreads()
        stride //= 2

    if tx == 0 and r < c1.shape[0] and sh_dE[0] != float32(0.0):
        cuda.atomic.add(E, r, sh_dE[0])


@cuda.jit
def tbg_helicity_sums_kernel(
    c1,
    s1,
    c2,
    s2,
    amp1_sq,
    sum_cos_x,
    sum_sin_x,
    sum_cos_y,
    sum_sin_y,
):
    """
    Accumulate phase-sum helicity sums with amplitude-weighted bonds.
    """
    sh_cos_x = cuda.shared.array(shape=SHARED_REDUCTION_MAX_THREADS, dtype=float32)
    sh_sin_x = cuda.shared.array(shape=SHARED_REDUCTION_MAX_THREADS, dtype=float32)
    sh_cos_y = cuda.shared.array(shape=SHARED_REDUCTION_MAX_THREADS, dtype=float32)
    sh_sin_y = cuda.shared.array(shape=SHARED_REDUCTION_MAX_THREADS, dtype=float32)

    tx = cuda.threadIdx.x
    r = cuda.blockIdx.y
    L = c1.shape[2]
    half = c1.shape[3]
    area = 2 * L * half
    site = cuda.blockIdx.x * cuda.blockDim.x + tx

    local_cos_x = float32(0.0)
    local_sin_x = float32(0.0)
    local_cos_y = float32(0.0)
    local_sin_y = float32(0.0)

    if r < c1.shape[0] and site < area:
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

        x0 = amp1_sq[r, color, i, k]
        a10 = _amp1(x0)
        a20 = _amp2(x0)
        c10 = c1[r, color, i, k]
        s10 = s1[r, color, i, k]
        c20 = c2[r, color, i, k]
        s20 = s2[r, color, i, k]

        xx = amp1_sq[r, opp, ip, k]
        a1x = _amp1(xx)
        a2x = _amp2(xx)
        local_cos_x = a10 * a1x * _cos_delta(
            c10,
            s10,
            c1[r, opp, ip, k],
            s1[r, opp, ip, k],
        )
        local_cos_x += a20 * a2x * _cos_delta(
            c20,
            s20,
            c2[r, opp, ip, k],
            s2[r, opp, ip, k],
        )
        local_sin_x = a10 * a1x * _sin_delta(
            c10,
            s10,
            c1[r, opp, ip, k],
            s1[r, opp, ip, k],
        )
        local_sin_x += a20 * a2x * _sin_delta(
            c20,
            s20,
            c2[r, opp, ip, k],
            s2[r, opp, ip, k],
        )

        xy = amp1_sq[r, opp, i, kp]
        a1y = _amp1(xy)
        a2y = _amp2(xy)
        local_cos_y = a10 * a1y * _cos_delta(
            c10,
            s10,
            c1[r, opp, i, kp],
            s1[r, opp, i, kp],
        )
        local_cos_y += a20 * a2y * _cos_delta(
            c20,
            s20,
            c2[r, opp, i, kp],
            s2[r, opp, i, kp],
        )
        local_sin_y = a10 * a1y * _sin_delta(
            c10,
            s10,
            c1[r, opp, i, kp],
            s1[r, opp, i, kp],
        )
        local_sin_y += a20 * a2y * _sin_delta(
            c20,
            s20,
            c2[r, opp, i, kp],
            s2[r, opp, i, kp],
        )

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

    if tx == 0 and r < c1.shape[0]:
        cuda.atomic.add(sum_cos_x, r, sh_cos_x[0])
        cuda.atomic.add(sum_sin_x, r, sh_sin_x[0])
        cuda.atomic.add(sum_cos_y, r, sh_cos_y[0])
        cuda.atomic.add(sum_sin_y, r, sh_sin_y[0])


@cuda.jit
def tbg_z2_and_amplitude_sums_kernel(
    c1,
    s1,
    c2,
    s2,
    amp1_sq,
    z2_sum,
    amp_imbalance_sum,
):
    """
    Accumulate Z2 chirality and relative-density/amplitude observables.
    """
    sh_z2 = cuda.shared.array(shape=SHARED_REDUCTION_MAX_THREADS, dtype=float32)
    sh_amp = cuda.shared.array(shape=SHARED_REDUCTION_MAX_THREADS, dtype=float32)

    tx = cuda.threadIdx.x
    r = cuda.blockIdx.y
    L = c1.shape[2]
    half = c1.shape[3]
    area = 2 * L * half
    site = cuda.blockIdx.x * cuda.blockDim.x + tx

    local_z2 = float32(0.0)
    local_amp = float32(0.0)

    if r < c1.shape[0] and site < area:
        color = site // (L * half)
        rem = site - color * L * half
        i = rem // half
        k = rem - i * half
        rel_sin = _sin_delta(
            c1[r, color, i, k],
            s1[r, color, i, k],
            c2[r, color, i, k],
            s2[r, color, i, k],
        )
        if rel_sin >= float32(0.0):
            local_z2 = float32(1.0)
        else:
            local_z2 = float32(-1.0)
        local_amp = float32(2.0) * amp1_sq[r, color, i, k] - float32(1.0)

    sh_z2[tx] = local_z2
    sh_amp[tx] = local_amp
    cuda.syncthreads()

    stride = cuda.blockDim.x // 2
    while stride > 0:
        if tx < stride:
            sh_z2[tx] += sh_z2[tx + stride]
            sh_amp[tx] += sh_amp[tx + stride]
        cuda.syncthreads()
        stride //= 2

    if tx == 0 and r < c1.shape[0]:
        cuda.atomic.add(z2_sum, r, sh_z2[0])
        cuda.atomic.add(amp_imbalance_sum, r, sh_amp[0])


@cuda.jit
def record_helicity_by_slot_kernel(
    sum_cos_x,
    sum_sin_x,
    sum_cos_y,
    sum_sin_y,
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
    Kx = sum_cos_x[walker]
    Ix = sum_sin_x[walker]
    Ky = sum_cos_y[walker]
    Iy = sum_sin_y[walker]
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
    Kx = sum_cos_x[walker]
    Ix = sum_sin_x[walker]
    Ky = sum_cos_y[walker]
    Iy = sum_sin_y[walker]
    Kx_block_sums[slot, block] += Kx
    Ix_block_sums[slot, block] += Ix
    Ix2_block_sums[slot, block] += Ix * Ix
    Ky_block_sums[slot, block] += Ky
    Iy_block_sums[slot, block] += Iy
    Iy2_block_sums[slot, block] += Iy * Iy


@dataclass(frozen=True)
class TBGModel(BaseModel):
    """
    Two-component TBG Ginzburg-Landau lattice model with amplitude fluctuations.

    The local state is Delta_a = |Delta_a| exp(i phi_a) with
    |Delta_1|^2 + |Delta_2|^2 = 1, represented by x = |Delta_1|^2.
    """

    K: float = 1.0
    phase_step: float = math.pi / 2.0
    amplitude_step: float = 0.1
    ordered_start: bool = False
    name: str = "tbg"
    output_prefix: str = "tbg2d"

    def kernel_K(self) -> np.float32:
        return np.float32(self.K)

    def max_threads_per_block(self) -> int:
        return SHARED_REDUCTION_MAX_THREADS

    def validate_lattice(self, L: int) -> None:
        super().validate_lattice(L)
        if int(L) % 2 != 0:
            raise ValueError("TBGModel requires even L for two-color updates.")

    def update_sites_per_walker(self, L: int) -> int:
        self.validate_lattice(L)
        return int(L) * (int(L) // 2)

    def metadata(self) -> dict[str, Any]:
        return {
            "model_name": self.name,
            "output_prefix": self.output_prefix,
            "K": float(self.K),
            "phase_step": float(self.phase_step),
            "amplitude_step": float(self.amplitude_step),
            "ordered_start": bool(self.ordered_start),
            "hamiltonian": (
                "H = -sum_<ij>,a |Delta_ai||Delta_aj| cos(phi_aj - phi_ai) "
                "+ sum_i K |Delta_1i|^2 |Delta_2i|^2 "
                "[cos(2(phi_1i - phi_2i)) - 1], "
                "|Delta_1i|^2 + |Delta_2i|^2 = 1"
            ),
            "reference": (
                "Phys. Rev. B 107, 064501 (2023), supplemental Eq. (1)"
            ),
            "reference_doi": "10.1103/PhysRevB.107.064501",
            "update_scheme": (
                "two-color checkerboard Metropolis updates of phi_1, phi_2, "
                "and x = |Delta_1|^2"
            ),
            "derived_observables": [
                "z2_chirality",
                "z2_binder_ratio",
                "relative_density_amplitude",
                "u1_phase_sum_helicity_modulus",
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
    ) -> "TBGRuntime":
        return TBGRuntime(
            model=self,
            L=L,
            R=R,
            rng=rng,
            phase_step=self.phase_step,
            amplitude_step=self.amplitude_step,
            threads_per_block=threads_per_block,
            full_site_blocks=full_site_blocks,
            update_blocks_per_walker=update_blocks_per_walker,
            slot_blocks=slot_blocks,
            full_lattice_blocks_per_walker=full_lattice_blocks_per_walker,
            inv_N=inv_N,
        )


class TBGRuntime(BaseRuntime):
    """
    Live CUDA state for TBGModel.
    """

    def __init__(
        self,
        *,
        model: TBGModel,
        L: int,
        R: int,
        rng: np.random.Generator,
        phase_step: float,
        amplitude_step: float,
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
            raise ValueError("TBGRuntime requires at least one walker.")
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
        self.phase_step = np.float32(phase_step)
        self.amplitude_step = np.float32(amplitude_step)
        if not np.isfinite(self.phase_step) or self.phase_step <= 0.0:
            raise ValueError("phase_step must be finite and positive.")
        if not np.isfinite(self.amplitude_step) or self.amplitude_step <= 0.0:
            raise ValueError("amplitude_step must be finite and positive.")
        if not np.isfinite(self.K) or self.K < 0.0:
            raise ValueError("K must be finite and nonnegative.")

        if self.model.ordered_start:
            phi1_h = np.full((self.R, self.L, self.L), float(HALF_PI), dtype=np.float32)
            phi2_h = np.zeros((self.R, self.L, self.L), dtype=np.float32)
            amp1_sq_h = np.full((self.R, self.L, self.L), 0.5, dtype=np.float32)
        else:
            phi1_h = rng.uniform(
                0.0,
                float(TWO_PI),
                size=(self.R, self.L, self.L),
            ).astype(np.float32)
            phi2_h = rng.uniform(
                0.0,
                float(TWO_PI),
                size=(self.R, self.L, self.L),
            ).astype(np.float32)
            amp1_sq_h = rng.uniform(
                0.0,
                1.0,
                size=(self.R, self.L, self.L),
            ).astype(np.float32)

        self.d_c1 = cuda.to_device(
            pack_two_color_checkerboard(np.cos(phi1_h).astype(np.float32))
        )
        self.d_s1 = cuda.to_device(
            pack_two_color_checkerboard(np.sin(phi1_h).astype(np.float32))
        )
        self.d_c2 = cuda.to_device(
            pack_two_color_checkerboard(np.cos(phi2_h).astype(np.float32))
        )
        self.d_s2 = cuda.to_device(
            pack_two_color_checkerboard(np.sin(phi2_h).astype(np.float32))
        )
        self.d_amp1_sq = cuda.to_device(pack_two_color_checkerboard(amp1_sq_h))

        self.d_E = cuda.device_array(self.R, dtype=np.float32)
        self.d_E_recomputed = cuda.device_array(self.R, dtype=np.float32)
        self.d_sum_cos_x = cuda.device_array(self.R, dtype=np.float32)
        self.d_sum_sin_x = cuda.device_array(self.R, dtype=np.float32)
        self.d_sum_cos_y = cuda.device_array(self.R, dtype=np.float32)
        self.d_sum_sin_y = cuda.device_array(self.R, dtype=np.float32)
        self.d_z2_sum = cuda.device_array(self.R, dtype=np.float32)
        self.d_amplitude_imbalance_sum = cuda.device_array(self.R, dtype=np.float32)

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
        self.d_amplitude_update_attempts = cuda.to_device(
            np.zeros(self.R, dtype=np.int64)
        )
        self.d_amplitude_update_acceptance = cuda.to_device(
            np.zeros(self.R, dtype=np.int64)
        )

        self.d_energies = None
        self.d_energy_block_sums = None
        self.d_energy2_block_sums = None
        self.d_z2_order_parameter = None
        self.d_amplitude_imbalance = None
        self.d_helicities = None
        self.d_z2_abs_block_sums = None
        self.d_z2_2_block_sums = None
        self.d_z2_4_block_sums = None
        self.d_amp_abs_block_sums = None
        self.d_amp2_block_sums = None
        self.d_amp4_block_sums = None
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
        tbg_energy_init_kernel[self.full_site_blocks, self.threads_per_block](
            self.d_c1,
            self.d_s1,
            self.d_c2,
            self.d_s2,
            self.d_amp1_sq,
            self.K,
            self.d_E,
        )

    def sweep(self, betas_by_walker, rng_states_updates, slot_of_walker) -> None:
        del slot_of_walker
        for color in (0, 1):
            tbg_update_kernel[
                (self.update_blocks_per_walker, self.R),
                self.threads_per_block,
            ](
                self.d_c1,
                self.d_s1,
                self.d_c2,
                self.d_s2,
                self.d_amp1_sq,
                betas_by_walker,
                rng_states_updates,
                color,
                self.phase_step,
                self.amplitude_step,
                self.K,
                self.d_E,
                self.d_local_update_attempts,
                self.d_local_update_acceptance,
                self.d_phase_update_attempts,
                self.d_phase_update_acceptance,
                self.d_amplitude_update_attempts,
                self.d_amplitude_update_acceptance,
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
            self.d_amplitude_update_attempts,
            self.d_amplitude_update_acceptance,
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
        tbg_energy_init_kernel[self.full_site_blocks, self.threads_per_block](
            self.d_c1,
            self.d_s1,
            self.d_c2,
            self.d_s2,
            self.d_amp1_sq,
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
            self.d_z2_order_parameter = cuda.device_array(
                (self.R, n_derived_meas),
                dtype=np.float32,
            )
            self.d_amplitude_imbalance = cuda.device_array(
                (self.R, n_derived_meas),
                dtype=np.float32,
            )
            self.d_helicities = cuda.device_array(
                (self.R, n_derived_meas),
                dtype=np.float32,
            )
        else:
            self.d_z2_order_parameter = None
            self.d_amplitude_imbalance = None
            self.d_helicities = None

        n_d_blocks, d_block_size = block_count_and_size(
            n_derived_meas,
            observable_n_blocks,
        )
        self.derived_observable_block_size = np.int32(d_block_size)
        self.helicity_observable_block_size = np.int32(d_block_size)
        if n_d_blocks > 0:
            zeros = np.zeros((self.R, n_d_blocks), dtype=np.float32)
            self.d_z2_abs_block_sums = cuda.to_device(zeros)
            self.d_z2_2_block_sums = cuda.to_device(zeros)
            self.d_z2_4_block_sums = cuda.to_device(zeros)
            self.d_amp_abs_block_sums = cuda.to_device(zeros)
            self.d_amp2_block_sums = cuda.to_device(zeros)
            self.d_amp4_block_sums = cuda.to_device(zeros)
            self.d_helicity_Kx_block_sums = cuda.to_device(zeros)
            self.d_helicity_Ix_block_sums = cuda.to_device(zeros)
            self.d_helicity_Ix2_block_sums = cuda.to_device(zeros)
            self.d_helicity_Ky_block_sums = cuda.to_device(zeros)
            self.d_helicity_Iy_block_sums = cuda.to_device(zeros)
            self.d_helicity_Iy2_block_sums = cuda.to_device(zeros)
        else:
            self.d_z2_abs_block_sums = None
            self.d_z2_2_block_sums = None
            self.d_z2_4_block_sums = None
            self.d_amp_abs_block_sums = None
            self.d_amp2_block_sums = None
            self.d_amp4_block_sums = None
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
        tbg_helicity_sums_kernel[
            (self.full_lattice_blocks_per_walker, self.R),
            self.threads_per_block,
        ](
            self.d_c1,
            self.d_s1,
            self.d_c2,
            self.d_s2,
            self.d_amp1_sq,
            self.d_sum_cos_x,
            self.d_sum_sin_x,
            self.d_sum_cos_y,
            self.d_sum_sin_y,
        )

    def _compute_z2_and_amplitude_sums(self) -> None:
        fill_two_vectors_kernel[self.slot_blocks, self.threads_per_block](
            self.d_z2_sum,
            self.d_amplitude_imbalance_sum,
            0.0,
        )
        tbg_z2_and_amplitude_sums_kernel[
            (self.full_lattice_blocks_per_walker, self.R),
            self.threads_per_block,
        ](
            self.d_c1,
            self.d_s1,
            self.d_c2,
            self.d_s2,
            self.d_amp1_sq,
            self.d_z2_sum,
            self.d_amplitude_imbalance_sum,
        )

    def record_derived_observables(
        self,
        betas_by_walker,
        walker_of_slot,
        col: int,
    ) -> None:
        if self.d_helicities is None:
            return
        self._compute_z2_and_amplitude_sums()
        self._compute_helicity_sums()

        record_scalar_by_slot_kernel[self.slot_blocks, self.threads_per_block](
            self.d_z2_sum,
            walker_of_slot,
            self.d_z2_order_parameter,
            col,
        )
        record_scalar_by_slot_kernel[self.slot_blocks, self.threads_per_block](
            self.d_amplitude_imbalance_sum,
            walker_of_slot,
            self.d_amplitude_imbalance,
            col,
        )
        record_helicity_by_slot_kernel[self.slot_blocks, self.threads_per_block](
            self.d_sum_cos_x,
            self.d_sum_sin_x,
            self.d_sum_cos_y,
            self.d_sum_sin_y,
            betas_by_walker,
            walker_of_slot,
            self.inv_N,
            self.d_helicities,
            col,
        )

        if self.d_z2_abs_block_sums is not None:
            accumulate_order_block_moments_by_slot_kernel[
                self.slot_blocks,
                self.threads_per_block,
            ](
                self.d_z2_sum,
                walker_of_slot,
                self.d_z2_abs_block_sums,
                self.d_z2_2_block_sums,
                self.d_z2_4_block_sums,
                col,
                int(self.derived_observable_block_size),
            )
            accumulate_order_block_moments_by_slot_kernel[
                self.slot_blocks,
                self.threads_per_block,
            ](
                self.d_amplitude_imbalance_sum,
                walker_of_slot,
                self.d_amp_abs_block_sums,
                self.d_amp2_block_sums,
                self.d_amp4_block_sums,
                col,
                int(self.derived_observable_block_size),
            )
            accumulate_helicity_block_moments_by_slot_kernel[
                self.slot_blocks,
                self.threads_per_block,
            ](
                self.d_sum_cos_x,
                self.d_sum_sin_x,
                self.d_sum_cos_y,
                self.d_sum_sin_y,
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
        z2_order_parameter = (
            self.d_z2_order_parameter.copy_to_host()
            if self.d_z2_order_parameter is not None
            else np.empty((self.R, 0), dtype=np.float32)
        )
        amplitude_imbalance = (
            self.d_amplitude_imbalance.copy_to_host()
            if self.d_amplitude_imbalance is not None
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
        if self.d_z2_abs_block_sums is not None and d_block_size > 0:
            inv_d_block_size = np.float32(1.0 / float(d_block_size))
            z2_abs_block_means = (
                self.d_z2_abs_block_sums.copy_to_host() * inv_d_block_size
            )
            z2_2_block_means = (
                self.d_z2_2_block_sums.copy_to_host() * inv_d_block_size
            )
            z2_4_block_means = (
                self.d_z2_4_block_sums.copy_to_host() * inv_d_block_size
            )
            amplitude_abs_block_means = (
                self.d_amp_abs_block_sums.copy_to_host() * inv_d_block_size
            )
            amplitude2_block_means = (
                self.d_amp2_block_sums.copy_to_host() * inv_d_block_size
            )
            amplitude4_block_means = (
                self.d_amp4_block_sums.copy_to_host() * inv_d_block_size
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
            z2_abs_block_means = np.empty((self.R, 0), dtype=np.float32)
            z2_2_block_means = np.empty((self.R, 0), dtype=np.float32)
            z2_4_block_means = np.empty((self.R, 0), dtype=np.float32)
            amplitude_abs_block_means = np.empty((self.R, 0), dtype=np.float32)
            amplitude2_block_means = np.empty((self.R, 0), dtype=np.float32)
            amplitude4_block_means = np.empty((self.R, 0), dtype=np.float32)
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
            "z2_order_parameter": z2_order_parameter,
            "order_parameter": z2_order_parameter,
            "amplitude_imbalance": amplitude_imbalance,
            "helicities": helicities,
            "z2_abs_block_means": z2_abs_block_means,
            "z2_2_block_means": z2_2_block_means,
            "z2_4_block_means": z2_4_block_means,
            "order_abs_block_means": z2_abs_block_means,
            "order2_block_means": z2_2_block_means,
            "order4_block_means": z2_4_block_means,
            "amplitude_abs_block_means": amplitude_abs_block_means,
            "amplitude2_block_means": amplitude2_block_means,
            "amplitude4_block_means": amplitude4_block_means,
            "local_update_attempts": self.d_local_update_attempts.copy_to_host(),
            "local_update_acceptance": self.d_local_update_acceptance.copy_to_host(),
            "phase_update_attempts": self.d_phase_update_attempts.copy_to_host(),
            "phase_update_acceptance": self.d_phase_update_acceptance.copy_to_host(),
            "amplitude_update_attempts": (
                self.d_amplitude_update_attempts.copy_to_host()
            ),
            "amplitude_update_acceptance": (
                self.d_amplitude_update_acceptance.copy_to_host()
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


__all__ = ["TBGModel", "TBGRuntime"]
