from __future__ import annotations

from dataclasses import dataclass
import math

from numba import njit, prange
import numpy as np

from ptmc.cpu.interface import BaseCPUModel


@njit(parallel=True)
def _xy_sweep_walkers_numba(
    states: np.ndarray,
    betas_by_walker: np.ndarray,
    energy_by_walker: np.ndarray,
    sites_by_walker: np.ndarray,
    accept_randoms: np.ndarray,
    proposal_randoms: np.ndarray,
    local_update_attempts: np.ndarray,
    local_update_acceptance: np.ndarray,
    J: float,
    theta_step: float,
) -> None:
    R = states.shape[0]
    L = states.shape[1]
    n_steps = sites_by_walker.shape[1]
    two_pi = 2.0 * math.pi
    for walker in prange(R):
        beta = betas_by_walker[walker]
        energy_delta = 0.0
        accepted_count = 0
        for step in range(n_steps):
            site = sites_by_walker[walker, step]
            i = site // L
            j = site - i * L
            ip = 0 if i + 1 == L else i + 1
            im = L - 1 if i == 0 else i - 1
            jp = 0 if j + 1 == L else j + 1
            jm = L - 1 if j == 0 else j - 1
            old = states[walker, i, j]
            new = old + (2.0 * proposal_randoms[walker, step] - 1.0) * theta_step

            old_sum = 0.0
            new_sum = 0.0
            neighbor = states[walker, ip, j]
            old_sum += math.cos(old - neighbor)
            new_sum += math.cos(new - neighbor)
            neighbor = states[walker, im, j]
            old_sum += math.cos(old - neighbor)
            new_sum += math.cos(new - neighbor)
            neighbor = states[walker, i, jp]
            old_sum += math.cos(old - neighbor)
            new_sum += math.cos(new - neighbor)
            neighbor = states[walker, i, jm]
            old_sum += math.cos(old - neighbor)
            new_sum += math.cos(new - neighbor)

            dE = J * (old_sum - new_sum)
            if dE <= 0.0 or accept_randoms[walker, step] < math.exp(-beta * dE):
                states[walker, i, j] = new % two_pi
                energy_delta += dE
                accepted_count += 1
        energy_by_walker[walker] += energy_delta
        local_update_attempts[walker] += n_steps
        local_update_acceptance[walker] += accepted_count


@dataclass(frozen=True)
class XYModel(BaseCPUModel):
    """
    Square-lattice XY model for the simple CPU benchmark runner.
    """

    J: float = 1.0
    theta_step: float = math.pi / 2.0
    ordered_start: bool = False
    name: str = "xy_cpu"
    output_prefix: str = "xy2d_cpu"
    needs_proposal_randoms = True

    def prepare_states(self, states: list[np.ndarray]) -> np.ndarray:
        return np.ascontiguousarray(np.stack(states, axis=0), dtype=np.float64)

    def sweep_walkers(
        self,
        *,
        states: np.ndarray,
        betas_by_walker: np.ndarray,
        energy_by_walker: np.ndarray,
        sites_by_walker: np.ndarray,
        accept_randoms: np.ndarray,
        proposal_randoms: np.ndarray | None,
        local_update_attempts: np.ndarray,
        local_update_acceptance: np.ndarray,
    ) -> None:
        if proposal_randoms is None:
            raise ValueError("XYModel requires proposal_randoms.")
        _xy_sweep_walkers_numba(
            states,
            betas_by_walker,
            energy_by_walker,
            sites_by_walker,
            accept_randoms,
            proposal_randoms,
            local_update_attempts,
            local_update_acceptance,
            float(self.J),
            float(self.theta_step),
        )

    def initial_state(
        self,
        L: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        self.validate_lattice(L)
        if self.ordered_start:
            return np.zeros((int(L), int(L)), dtype=np.float64)
        return rng.uniform(0.0, 2.0 * math.pi, size=(int(L), int(L)))

    def energy(self, state: np.ndarray) -> float:
        theta = np.asarray(state, dtype=np.float64)
        e_down = np.cos(theta - np.roll(theta, -1, axis=0))
        e_right = np.cos(theta - np.roll(theta, -1, axis=1))
        return float(-self.J * (np.sum(e_down) + np.sum(e_right)))

    def measure_observables(
        self,
        state: np.ndarray,
        beta: float,
    ) -> dict[str, float]:
        theta = np.asarray(state, dtype=np.float64)
        inv_N = 1.0 / float(theta.size)
        delta_x = theta - np.roll(theta, -1, axis=0)
        delta_y = theta - np.roll(theta, -1, axis=1)
        Kx = float(self.J * np.sum(np.cos(delta_x)))
        Ix = float(self.J * np.sum(np.sin(delta_x)))
        Ky = float(self.J * np.sum(np.cos(delta_y)))
        Iy = float(self.J * np.sum(np.sin(delta_y)))
        Yx = (Kx - float(beta) * Ix * Ix) * inv_N
        Yy = (Ky - float(beta) * Iy * Iy) * inv_N
        return {
            "helicity_Kx": Kx,
            "helicity_Ix": Ix,
            "helicity_Ix2": Ix * Ix,
            "helicity_Ky": Ky,
            "helicity_Iy": Iy,
            "helicity_Iy2": Iy * Iy,
            "helicity": 0.5 * (Yx + Yy),
        }

    def metadata(self) -> dict[str, object]:
        return {
            "model_name": self.name,
            "output_prefix": self.output_prefix,
            "J": float(self.J),
            "theta_step": float(self.theta_step),
            "ordered_start": bool(self.ordered_start),
            "hamiltonian": "H = -J sum_<ij> cos(theta_i - theta_j)",
            "update_scheme": "CPU random single-site Metropolis angle proposals",
            "derived_observables": ["u1_helicity_modulus", "bkt_intersection"],
            "analysis_capabilities": {
                "thermodynamics": True,
                "z2_order_parameter": False,
                "helicity_modulus": True,
            },
        }


__all__ = ["XYModel"]
