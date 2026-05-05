"""Physics kernels and runtime for the reduced chiral U(1) x Z2 model."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numba import cuda, float32
from numba.cuda.random import xoroshiro128p_uniform_float32

TWO_PI = np.float32(2.0 * math.pi)
SHARED_REDUCTION_MAX_THREADS = 512


@dataclass(frozen=True)
class ChiralU1Z2Model:
    """
    Host-side metadata for the reduced chiral U(1) x Z2 model.

    The CUDA device helpers immediately below implement the actual local
    Hamiltonian used by the kernels:

        H = sum_<ij> E_ij
        E_ij = - J * w(sigma_i, sigma_j) * cos(theta_i - theta_j)
        w = a + sigma_i * sigma_j
    """

    J: float = 1.0
    a: float = 1.0
    name: str = "reduced_chiral_u1_z2"
    output_prefix: str = "chiral_xy_gpu"

    def kernel_J(self) -> np.float32:
        return np.float32(self.J)

    def kernel_a(self) -> np.float32:
        return np.float32(self.a)

    def max_threads_per_block(self) -> int:
        return SHARED_REDUCTION_MAX_THREADS

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
    ) -> "ChiralU1Z2Runtime":
        return ChiralU1Z2Runtime(
            model=self,
            L=L,
            R=R,
            rng=rng,
            theta_step=theta_step,
            threads_per_block=threads_per_block,
            full_site_blocks=full_site_blocks,
            half_sweep_blocks_per_walker=half_sweep_blocks_per_walker,
            slot_blocks=slot_blocks,
            full_lattice_blocks_per_walker=full_lattice_blocks_per_walker,
            inv_N=inv_N,
        )

    def metadata(self) -> dict[str, object]:
        return {
            "model_name": self.name,
            "output_prefix": self.output_prefix,
            "J": float(self.J),
            "a": float(self.a),
            "bond_energy": (
                "E_ij = - J*w(sigma_i,sigma_j)*cos(theta_i-theta_j), "
                "w=a + sigma_i*sigma_j"
            ),
            "update_scheme": (
                "combined sigma-flip and theta-rotation sweeps with alternating order"
            ),
            "derived_observables": ["z2_magnetization", "u1_helicity_modulus"],
        }


@cuda.jit(device=True)
def model_cos_delta(c_i, st_i, c_j, st_j):
    """cos(theta_i - theta_j) from cached sine/cosine fields."""
    return c_i * c_j + st_i * st_j


@cuda.jit(device=True)
def model_sin_delta(c_i, st_i, c_j, st_j):
    """sin(theta_i - theta_j) from cached sine/cosine fields."""
    return st_i * c_j - c_i * st_j


@cuda.jit(device=True)
def model_phase_coupling_weight(sigma_i, sigma_j, a):
    """Theta-stiffness weight w(sigma_i, sigma_j) for one nearest-neighbor bond."""
    return a + float32(sigma_i) * float32(sigma_j)


@cuda.jit(device=True)
def model_bond_energy(sigma_i, c_i, st_i, sigma_j, c_j, st_j, J, a):
    """Nearest-neighbor bond energy E_ij for the current model."""
    weight = model_phase_coupling_weight(sigma_i, sigma_j, a)
    return - J * weight * model_cos_delta(c_i, st_i, c_j, st_j)


@cuda.jit(device=True)
def model_theta_delta_energy(
    sigma_i,
    c_old,
    st_old,
    c_new,
    st_new,
    sigma_j,
    c_j,
    st_j,
    J,
    a,
):
    """Energy change for changing theta_i while sigma_i is fixed."""
    weight = model_phase_coupling_weight(sigma_i, sigma_j, a)
    old_cos = model_cos_delta(c_old, st_old, c_j, st_j)
    new_cos = model_cos_delta(c_new, st_new, c_j, st_j)
    return J * weight * (old_cos - new_cos)


@cuda.jit(device=True)
def model_sigma_flip_delta_energy(sigma_i, c_i, st_i, sigma_j, c_j, st_j, J, a):
    """Energy change for flipping sigma_i while theta_i is fixed."""
    old_weight = model_phase_coupling_weight(sigma_i, sigma_j, a)
    new_weight = model_phase_coupling_weight(-sigma_i, sigma_j, a)
    cos_dtheta = model_cos_delta(c_i, st_i, c_j, st_j)
    return J * (old_weight - new_weight) * cos_dtheta


def random_state(L: int, rng: np.random.Generator, p_up: float = 0.5):
    """Create random sigma and theta fields on the host."""
    L = int(L)
    p_up = float(p_up)
    if L <= 0:
        raise ValueError("L must be positive.")
    if not (0.0 <= p_up <= 1.0):
        raise ValueError("p_up must lie between 0 and 1.")

    sigma = rng.choice(
        np.array([-1, 1], dtype=np.int8), size=(L, L), p=[1.0 - p_up, p_up]
    )
    theta = rng.uniform(0.0, TWO_PI, size=(L, L)).astype(np.float32)
    return sigma.astype(np.int8), theta


def _pack_checkerboard_field(field: np.ndarray) -> np.ndarray:
    """Pack a full (R, L, L) field into contiguous checkerboard colors."""
    if field.ndim != 3 or field.shape[1] != field.shape[2]:
        raise ValueError("Expected a field with shape (R, L, L).")

    R, L, _ = field.shape
    if L % 2 != 0:
        raise ValueError("Checkerboard packing requires even L.")

    half = L // 2
    packed = np.empty((R, 2, L, half), dtype=field.dtype)
    for color in (0, 1):
        for i in range(L):
            packed[:, color, i, :] = field[:, i, ((i + color) & 1) :: 2]
    return packed


@cuda.jit
def energy_init_kernel(sigmas, cos_thetas, sin_thetas, J, a, E_out):
    """Compute total energy with forward bonds only, accumulated per walker."""
    tid = cuda.grid(1)
    R = sigmas.shape[0]
    L = sigmas.shape[2]
    half = sigmas.shape[3]
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

    s0 = sigmas[r, color, i, k]
    c0 = cos_thetas[r, color, i, k]
    st0 = sin_thetas[r, color, i, k]

    sx = sigmas[r, opp, ip, k]
    ex = model_bond_energy(
        s0,
        c0,
        st0,
        sx,
        cos_thetas[r, opp, ip, k],
        sin_thetas[r, opp, ip, k],
        J,
        a,
    )

    sy = sigmas[r, opp, i, kp]
    ey = model_bond_energy(
        s0,
        c0,
        st0,
        sy,
        cos_thetas[r, opp, i, kp],
        sin_thetas[r, opp, i, kp],
        J,
        a,
    )

    cuda.atomic.add(E_out, r, ex + ey)


@cuda.jit
def magnetization_init_kernel(sigmas, M_out):
    """Compute total Z2 magnetization per walker."""
    tid = cuda.grid(1)
    R = sigmas.shape[0]
    L = sigmas.shape[2]
    half = sigmas.shape[3]
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
    cuda.atomic.add(M_out, r, float32(sigmas[r, color, i, k]))


@cuda.jit
def theta_update_kernel(
    sigmas,
    cos_thetas,
    sin_thetas,
    betas_by_walker,
    rng_states,
    color,
    theta_step,
    J,
    a,
    E,
):
    """One checkerboard half-sweep of theta proposals only."""
    sh_dE = cuda.shared.array(shape=SHARED_REDUCTION_MAX_THREADS, dtype=float32)

    tx = cuda.threadIdx.x
    r = cuda.blockIdx.y
    L = sigmas.shape[2]
    half = sigmas.shape[3]
    sites_per_replica = L * half
    site_idx = cuda.blockIdx.x * cuda.blockDim.x + tx

    dE_acc = float32(0.0)

    if r < sigmas.shape[0] and site_idx < sites_per_replica:
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
        spin0 = sigmas[r, color, i, k]
        c0 = cos_thetas[r, color, i, k]
        st0 = sin_thetas[r, color, i, k]
        rng_idx = r * sites_per_replica + site_idx

        u = xoroshiro128p_uniform_float32(rng_states, rng_idx)
        dtheta = theta_step * (float32(2.0) * u - float32(1.0))
        sin_d = float32(math.sin(dtheta))
        cos_d = float32(math.cos(dtheta))
        c_new = c0 * cos_d - st0 * sin_d
        st_new = st0 * cos_d + c0 * sin_d
        norm2 = c_new * c_new + st_new * st_new
        inv_norm = float32(1.0) / float32(math.sqrt(norm2))
        c_new *= inv_norm
        st_new *= inv_norm

        dE = float32(0.0)

        spin_j = sigmas[r, opp, ip, k]
        dE += model_theta_delta_energy(
            spin0,
            c0,
            st0,
            c_new,
            st_new,
            spin_j,
            cos_thetas[r, opp, ip, k],
            sin_thetas[r, opp, ip, k],
            J,
            a,
        )

        spin_j = sigmas[r, opp, im, k]
        dE += model_theta_delta_energy(
            spin0,
            c0,
            st0,
            c_new,
            st_new,
            spin_j,
            cos_thetas[r, opp, im, k],
            sin_thetas[r, opp, im, k],
            J,
            a,
        )

        spin_j = sigmas[r, opp, i, kp]
        dE += model_theta_delta_energy(
            spin0,
            c0,
            st0,
            c_new,
            st_new,
            spin_j,
            cos_thetas[r, opp, i, kp],
            sin_thetas[r, opp, i, kp],
            J,
            a,
        )

        spin_j = sigmas[r, opp, i, km]
        dE += model_theta_delta_energy(
            spin0,
            c0,
            st0,
            c_new,
            st_new,
            spin_j,
            cos_thetas[r, opp, i, km],
            sin_thetas[r, opp, i, km],
            J,
            a,
        )

        accepted = dE <= float32(0.0)
        if not accepted:
            acc = xoroshiro128p_uniform_float32(rng_states, rng_idx)
            accepted = acc < float32(math.exp(-(beta * dE)))

        if accepted:
            cos_thetas[r, color, i, k] = c_new
            sin_thetas[r, color, i, k] = st_new
            dE_acc = dE

    sh_dE[tx] = dE_acc
    cuda.syncthreads()

    stride = cuda.blockDim.x // 2
    while stride > 0:
        if tx < stride:
            sh_dE[tx] += sh_dE[tx + stride]
        cuda.syncthreads()
        stride //= 2

    if tx == 0 and r < sigmas.shape[0]:
        if sh_dE[0] != float32(0.0):
            cuda.atomic.add(E, r, sh_dE[0])


@cuda.jit
def sigma_update_kernel(
    sigmas,
    cos_thetas,
    sin_thetas,
    betas_by_walker,
    rng_states,
    color,
    J,
    a,
    E,
    M,
):
    """One checkerboard half-sweep of sigma-flip proposals only."""
    sh_dE = cuda.shared.array(shape=SHARED_REDUCTION_MAX_THREADS, dtype=float32)
    sh_dM = cuda.shared.array(shape=SHARED_REDUCTION_MAX_THREADS, dtype=float32)

    tx = cuda.threadIdx.x
    r = cuda.blockIdx.y
    L = sigmas.shape[2]
    half = sigmas.shape[3]
    sites_per_replica = L * half
    site_idx = cuda.blockIdx.x * cuda.blockDim.x + tx

    dE_acc = float32(0.0)
    dM_acc = float32(0.0)

    if r < sigmas.shape[0] and site_idx < sites_per_replica:
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
        spin0 = sigmas[r, color, i, k]
        c0 = cos_thetas[r, color, i, k]
        st0 = sin_thetas[r, color, i, k]
        rng_idx = r * sites_per_replica + site_idx

        dE = float32(0.0)

        spin_j = sigmas[r, opp, ip, k]
        dE += model_sigma_flip_delta_energy(
            spin0,
            c0,
            st0,
            spin_j,
            cos_thetas[r, opp, ip, k],
            sin_thetas[r, opp, ip, k],
            J,
            a,
        )

        spin_j = sigmas[r, opp, im, k]
        dE += model_sigma_flip_delta_energy(
            spin0,
            c0,
            st0,
            spin_j,
            cos_thetas[r, opp, im, k],
            sin_thetas[r, opp, im, k],
            J,
            a,
        )

        spin_j = sigmas[r, opp, i, kp]
        dE += model_sigma_flip_delta_energy(
            spin0,
            c0,
            st0,
            spin_j,
            cos_thetas[r, opp, i, kp],
            sin_thetas[r, opp, i, kp],
            J,
            a,
        )

        spin_j = sigmas[r, opp, i, km]
        dE += model_sigma_flip_delta_energy(
            spin0,
            c0,
            st0,
            spin_j,
            cos_thetas[r, opp, i, km],
            sin_thetas[r, opp, i, km],
            J,
            a,
        )

        accepted = dE <= float32(0.0)
        if not accepted:
            acc = xoroshiro128p_uniform_float32(rng_states, rng_idx)
            accepted = acc < float32(math.exp(-(beta * dE)))

        if accepted:
            sigmas[r, color, i, k] = -spin0
            dE_acc = dE
            dM_acc = -float32(2.0) * float32(spin0)

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

    if tx == 0 and r < sigmas.shape[0]:
        if sh_dE[0] != float32(0.0):
            cuda.atomic.add(E, r, sh_dE[0])
        if sh_dM[0] != float32(0.0):
            cuda.atomic.add(M, r, sh_dM[0])


@cuda.jit
def helicity_sums_kernel(
    sigmas,
    cos_thetas,
    sin_thetas,
    a,
    sum_cos_x,
    sum_sin_x,
    sum_cos_y,
    sum_sin_y,
):
    """Accumulate helicity sums per walker with block-local reduction."""
    sh_cos_x = cuda.shared.array(shape=SHARED_REDUCTION_MAX_THREADS, dtype=float32)
    sh_sin_x = cuda.shared.array(shape=SHARED_REDUCTION_MAX_THREADS, dtype=float32)
    sh_cos_y = cuda.shared.array(shape=SHARED_REDUCTION_MAX_THREADS, dtype=float32)
    sh_sin_y = cuda.shared.array(shape=SHARED_REDUCTION_MAX_THREADS, dtype=float32)

    tx = cuda.threadIdx.x
    r = cuda.blockIdx.y
    L = sigmas.shape[2]
    half = sigmas.shape[3]
    area = 2 * L * half
    site = cuda.blockIdx.x * cuda.blockDim.x + tx

    local_cos_x = float32(0.0)
    local_sin_x = float32(0.0)
    local_cos_y = float32(0.0)
    local_sin_y = float32(0.0)

    if r < sigmas.shape[0] and site < area:
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

        s0 = sigmas[r, color, i, k]
        c0 = cos_thetas[r, color, i, k]
        st0 = sin_thetas[r, color, i, k]

        sx_sigma = sigmas[r, opp, ip, k]
        cx = cos_thetas[r, opp, ip, k]
        sx = sin_thetas[r, opp, ip, k]
        weight_x = model_phase_coupling_weight(s0, sx_sigma, a)
        if weight_x != float32(0.0):
            local_cos_x = weight_x * model_cos_delta(c0, st0, cx, sx)
            local_sin_x = weight_x * model_sin_delta(c0, st0, cx, sx)

        sy_sigma = sigmas[r, opp, i, kp]
        cy = cos_thetas[r, opp, i, kp]
        sy = sin_thetas[r, opp, i, kp]
        weight_y = model_phase_coupling_weight(s0, sy_sigma, a)
        if weight_y != float32(0.0):
            local_cos_y = weight_y * model_cos_delta(c0, st0, cy, sy)
            local_sin_y = weight_y * model_sin_delta(c0, st0, cy, sy)

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

    if tx == 0 and r < sigmas.shape[0]:
        cuda.atomic.add(sum_cos_x, r, sh_cos_x[0])
        cuda.atomic.add(sum_sin_x, r, sh_sin_x[0])
        cuda.atomic.add(sum_cos_y, r, sh_cos_y[0])
        cuda.atomic.add(sum_sin_y, r, sh_sin_y[0])


@cuda.jit
def record_energy_magnetization_by_slot_kernel(
    E_by_walker,
    M_by_walker,
    walker_of_slot,
    E_out,
    M_out,
    col,
):
    """Record energy and magnetization histories with one tiny kernel launch."""
    slot = cuda.grid(1)
    if slot < walker_of_slot.shape[0]:
        walker = walker_of_slot[slot]
        E_out[slot, col] = E_by_walker[walker]
        M_out[slot, col] = M_by_walker[walker]


@cuda.jit
def accumulate_observable_block_moments_by_slot_kernel(
    E_by_walker,
    M_by_walker,
    walker_of_slot,
    energy_block_sums,
    energy2_block_sums,
    mag_abs_block_sums,
    mag2_block_sums,
    mag4_block_sums,
    col,
    block_size,
):
    """Accumulate per-block observable moments in temperature-slot order."""
    slot = cuda.grid(1)
    if slot >= walker_of_slot.shape[0] or block_size <= 0:
        return

    block = col // block_size
    if block >= energy_block_sums.shape[1]:
        return

    walker = walker_of_slot[slot]
    E = E_by_walker[walker]
    M = M_by_walker[walker]
    M_abs = M if M >= float32(0.0) else -M
    E2 = E * E
    M2 = M * M

    energy_block_sums[slot, block] += E
    energy2_block_sums[slot, block] += E2
    mag_abs_block_sums[slot, block] += M_abs
    mag2_block_sums[slot, block] += M2
    mag4_block_sums[slot, block] += M2 * M2


@cuda.jit
def compute_and_record_helicity_by_slot_kernel(
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
    """Compute helicity by walker and write it directly in temperature-slot order."""
    slot = cuda.grid(1)
    if slot >= walker_of_slot.shape[0]:
        return

    walker = walker_of_slot[slot]
    beta = betas_by_walker[walker]
    pref = J * inv_N
    beta_pref = beta * J * J * inv_N
    sx = sum_sin_x[walker]
    sy = sum_sin_y[walker]
    Yx = pref * sum_cos_x[walker] - beta_pref * (sx * sx)
    Yy = pref * sum_cos_y[walker] - beta_pref * (sy * sy)
    out[slot, col] = 0.5 * (Yx + Yy)


@cuda.jit
def fill_two_vectors_kernel(a, b, value):
    """Fill two same-length device vectors in one launch."""
    idx = cuda.grid(1)
    if idx < a.shape[0]:
        a[idx] = value
        b[idx] = value


@cuda.jit
def fill_vector_kernel(a, value):
    """Fill one device vector."""
    idx = cuda.grid(1)
    if idx < a.shape[0]:
        a[idx] = value


@cuda.jit
def fill_four_vectors_kernel(a, b, c, d, value):
    """Fill four same-length device vectors in one launch."""
    idx = cuda.grid(1)
    if idx < a.shape[0]:
        a[idx] = value
        b[idx] = value
        c[idx] = value
        d[idx] = value


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
    """Replace running energies when periodic exact recomputation finds drift."""
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


class ChiralU1Z2Runtime:
    """
    GPU state and kernels for the reduced chiral U(1) x Z2 model.

    The parallel-tempering driver owns temperatures, replica labels, and run
    scheduling. This object owns everything that depends on the model fields,
    proposal schedule, Hamiltonian, or derived observables.
    """

    def __init__(
        self,
        *,
        model: ChiralU1Z2Model,
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
    ):
        self.model = model
        self.L = int(L)
        self.R = int(R)
        self.threads_per_block = int(threads_per_block)
        if self.L <= 0 or self.L % 2 != 0:
            raise ValueError("ChiralU1Z2Runtime requires a positive even L.")
        if self.R <= 0:
            raise ValueError("ChiralU1Z2Runtime requires at least one walker.")
        if self.threads_per_block <= 0:
            raise ValueError("threads_per_block must be positive.")
        if self.threads_per_block > SHARED_REDUCTION_MAX_THREADS:
            raise ValueError(
                f"threads_per_block exceeds {SHARED_REDUCTION_MAX_THREADS}."
            )
        if self.threads_per_block & (self.threads_per_block - 1):
            raise ValueError("threads_per_block must be a power of two.")

        self.full_site_blocks = int(full_site_blocks)
        self.half_sweep_blocks_per_walker = int(half_sweep_blocks_per_walker)
        self.slot_blocks = int(slot_blocks)
        self.full_lattice_blocks_per_walker = int(full_lattice_blocks_per_walker)
        self.inv_N = np.float32(inv_N)
        self.J = self.model.kernel_J()
        self.a = self.model.kernel_a()
        self.theta_step = np.float32(theta_step)
        if not np.isfinite(self.theta_step) or self.theta_step <= 0.0:
            raise ValueError("theta_step must be finite and positive.")
        self._next_update_is_sigma = True

        sigmas_h = np.empty((self.R, self.L, self.L), dtype=np.int8)
        thetas_h = np.empty((self.R, self.L, self.L), dtype=np.float32)
        for r in range(self.R):
            sigma_r, theta_r = random_state(self.L, rng)
            sigmas_h[r] = sigma_r
            thetas_h[r] = theta_r

        self.d_sigmas = cuda.to_device(_pack_checkerboard_field(sigmas_h))
        self.d_cos_thetas = cuda.to_device(
            _pack_checkerboard_field(np.cos(thetas_h).astype(np.float32))
        )
        self.d_sin_thetas = cuda.to_device(
            _pack_checkerboard_field(np.sin(thetas_h).astype(np.float32))
        )

        self.d_E = cuda.device_array(self.R, dtype=np.float32)
        self.d_E_recomputed = cuda.device_array(self.R, dtype=np.float32)
        self.d_M = cuda.device_array(self.R, dtype=np.float32)
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

        self.d_energies = None
        self.d_mags = None
        self.d_helicities = None
        self.d_energy_block_sums = None
        self.d_energy2_block_sums = None
        self.d_mag_abs_block_sums = None
        self.d_mag2_block_sums = None
        self.d_mag4_block_sums = None
        self.energies = None
        self.mags = None
        self.helicities = None
        self.energy_block_means = None
        self.energy2_block_means = None
        self.mag_abs_block_means = None
        self.mag2_block_means = None
        self.mag4_block_means = None
        self.observable_block_size = np.int32(0)
        self.energy_drift_last = np.zeros(self.R, dtype=np.float32)
        self.energy_drift_max = np.zeros(self.R, dtype=np.float32)
        self.energy_recompute_checks = np.zeros(self.R, dtype=np.int64)
        self.energy_recompute_corrections = np.zeros(self.R, dtype=np.int64)

        self._initialize_observables()

    @property
    def energy_by_walker(self):
        return self.d_E

    def _initialize_observables(self):
        fill_two_vectors_kernel[self.slot_blocks, self.threads_per_block](
            self.d_E,
            self.d_M,
            0.0,
        )
        energy_init_kernel[self.full_site_blocks, self.threads_per_block](
            self.d_sigmas,
            self.d_cos_thetas,
            self.d_sin_thetas,
            self.J,
            self.a,
            self.d_E,
        )
        magnetization_init_kernel[self.full_site_blocks, self.threads_per_block](
            self.d_sigmas,
            self.d_M,
        )

    def maybe_recompute_energy(
        self,
        sweeps_completed: int,
        recompute_stride: int,
        tolerance_per_site: np.float32,
    ):
        if recompute_stride <= 0:
            return
        if sweeps_completed % recompute_stride != 0:
            return

        fill_vector_kernel[self.slot_blocks, self.threads_per_block](
            self.d_E_recomputed,
            0.0,
        )
        energy_init_kernel[self.full_site_blocks, self.threads_per_block](
            self.d_sigmas,
            self.d_cos_thetas,
            self.d_sin_thetas,
            self.J,
            self.a,
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

    def _sweep_sigma(self, betas_by_walker, rng_states_updates):
        for color in (0, 1):
            sigma_update_kernel[
                (self.half_sweep_blocks_per_walker, self.R), self.threads_per_block
            ](
                self.d_sigmas,
                self.d_cos_thetas,
                self.d_sin_thetas,
                betas_by_walker,
                rng_states_updates,
                color,
                self.J,
                self.a,
                self.d_E,
                self.d_M,
            )

    def _sweep_theta(self, betas_by_walker, rng_states_updates):
        for color in (0, 1):
            theta_update_kernel[
                (self.half_sweep_blocks_per_walker, self.R), self.threads_per_block
            ](
                self.d_sigmas,
                self.d_cos_thetas,
                self.d_sin_thetas,
                betas_by_walker,
                rng_states_updates,
                color,
                self.theta_step,
                self.J,
                self.a,
                self.d_E,
            )

    def sweep(self, betas_by_walker, rng_states_updates):
        if self._next_update_is_sigma:
            self._sweep_sigma(betas_by_walker, rng_states_updates)
            self._sweep_theta(betas_by_walker, rng_states_updates)
        else:
            self._sweep_theta(betas_by_walker, rng_states_updates)
            self._sweep_sigma(betas_by_walker, rng_states_updates)
        self._next_update_is_sigma = not self._next_update_is_sigma

    def allocate_measurement_storage(
        self,
        n_meas: int,
        n_derived_meas: int,
        store_primary_histories: bool,
        observable_n_blocks: int,
    ):
        n_meas = int(n_meas)
        if store_primary_histories:
            hist_shape = (self.R, n_meas)
            self.d_energies = cuda.device_array(hist_shape, dtype=np.float32)
            self.d_mags = cuda.device_array(hist_shape, dtype=np.float32)
        else:
            self.d_energies = None
            self.d_mags = None

        requested_blocks = max(0, int(observable_n_blocks))
        n_blocks = min(requested_blocks, n_meas // 2)
        if n_meas > 0 and requested_blocks > 0 and n_blocks < 2:
            n_blocks = 1
        block_size = n_meas // n_blocks if n_blocks > 0 else 0
        self.observable_block_size = np.int32(block_size)
        compact_shape = (self.R, n_blocks)
        if n_blocks > 0:
            zeros = np.zeros(compact_shape, dtype=np.float32)
            self.d_energy_block_sums = cuda.to_device(zeros)
            self.d_energy2_block_sums = cuda.to_device(zeros)
            self.d_mag_abs_block_sums = cuda.to_device(zeros)
            self.d_mag2_block_sums = cuda.to_device(zeros)
            self.d_mag4_block_sums = cuda.to_device(zeros)
        else:
            self.d_energy_block_sums = None
            self.d_energy2_block_sums = None
            self.d_mag_abs_block_sums = None
            self.d_mag2_block_sums = None
            self.d_mag4_block_sums = None

        helicity_shape = (self.R, int(n_derived_meas))
        self.d_helicities = (
            cuda.device_array(helicity_shape, dtype=np.float32)
            if n_derived_meas > 0
            else None
        )

    def record_primary_observables(self, walker_of_slot, col: int):
        if self.d_energies is not None and self.d_mags is not None:
            record_energy_magnetization_by_slot_kernel[
                self.slot_blocks, self.threads_per_block
            ](
                self.d_E,
                self.d_M,
                walker_of_slot,
                self.d_energies,
                self.d_mags,
                col,
            )
        if self.d_energy_block_sums is not None:
            accumulate_observable_block_moments_by_slot_kernel[
                self.slot_blocks, self.threads_per_block
            ](
                self.d_E,
                self.d_M,
                walker_of_slot,
                self.d_energy_block_sums,
                self.d_energy2_block_sums,
                self.d_mag_abs_block_sums,
                self.d_mag2_block_sums,
                self.d_mag4_block_sums,
                col,
                int(self.observable_block_size),
            )

    def _record_helicity(self, betas_by_walker, walker_of_slot, col: int):
        if self.d_helicities is None:
            return

        fill_four_vectors_kernel[self.slot_blocks, self.threads_per_block](
            self.d_sum_cos_x,
            self.d_sum_sin_x,
            self.d_sum_cos_y,
            self.d_sum_sin_y,
            0.0,
        )
        helicity_sums_kernel[
            (self.full_lattice_blocks_per_walker, self.R), self.threads_per_block
        ](
            self.d_sigmas,
            self.d_cos_thetas,
            self.d_sin_thetas,
            self.a,
            self.d_sum_cos_x,
            self.d_sum_sin_x,
            self.d_sum_cos_y,
            self.d_sum_sin_y,
        )
        compute_and_record_helicity_by_slot_kernel[
            self.slot_blocks, self.threads_per_block
        ](
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

    def record_derived_observables(self, betas_by_walker, walker_of_slot, col: int):
        self._record_helicity(betas_by_walker, walker_of_slot, col)

    def copy_measurements_to_host(self):
        self.energies = (
            self.d_energies.copy_to_host()
            if self.d_energies is not None
            else np.empty((self.R, 0), dtype=np.float32)
        )
        self.mags = (
            self.d_mags.copy_to_host()
            if self.d_mags is not None
            else np.empty((self.R, 0), dtype=np.float32)
        )
        self.helicities = (
            self.d_helicities.copy_to_host()
            if self.d_helicities is not None
            else np.empty((self.R, 0), dtype=np.float32)
        )
        block_size = int(self.observable_block_size)
        if self.d_energy_block_sums is not None and block_size > 0:
            inv_block_size = np.float32(1.0 / float(block_size))
            self.energy_block_means = (
                self.d_energy_block_sums.copy_to_host() * inv_block_size
            )
            self.energy2_block_means = (
                self.d_energy2_block_sums.copy_to_host() * inv_block_size
            )
            self.mag_abs_block_means = (
                self.d_mag_abs_block_sums.copy_to_host() * inv_block_size
            )
            self.mag2_block_means = (
                self.d_mag2_block_sums.copy_to_host() * inv_block_size
            )
            self.mag4_block_means = (
                self.d_mag4_block_sums.copy_to_host() * inv_block_size
            )
        else:
            empty = np.empty((self.R, 0), dtype=np.float32)
            self.energy_block_means = empty
            self.energy2_block_means = empty
            self.mag_abs_block_means = empty
            self.mag2_block_means = empty
            self.mag4_block_means = empty
        return {
            "energies": self.energies,
            "mags": self.mags,
            "helicities": self.helicities,
            "energy_block_means": self.energy_block_means,
            "energy2_block_means": self.energy2_block_means,
            "mag_abs_block_means": self.mag_abs_block_means,
            "mag2_block_means": self.mag2_block_means,
            "mag4_block_means": self.mag4_block_means,
            "observable_block_size": self.observable_block_size,
        }

    def sync_energy_drift_stats_from_gpu(self):
        self.d_energy_drift_last.copy_to_host(self.energy_drift_last)
        self.d_energy_drift_max.copy_to_host(self.energy_drift_max)
        self.d_energy_recompute_checks.copy_to_host(self.energy_recompute_checks)
        self.d_energy_recompute_corrections.copy_to_host(
            self.energy_recompute_corrections
        )
        return {
            "energy_drift_last": self.energy_drift_last,
            "energy_drift_max": self.energy_drift_max,
            "energy_recompute_checks": self.energy_recompute_checks,
            "energy_recompute_corrections": self.energy_recompute_corrections,
        }
