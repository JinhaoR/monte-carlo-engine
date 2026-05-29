from __future__ import annotations

from dataclasses import dataclass
import math

from numba import njit, prange
import numpy as np

from ptmc.cpu.interface import BaseCPUModel


@njit(parallel=True)
def _ising_sweep_walkers_numba(
    states: np.ndarray,
    betas_by_walker: np.ndarray,
    energy_by_walker: np.ndarray,
    sites_by_walker: np.ndarray,
    accept_randoms: np.ndarray,
    local_update_attempts: np.ndarray,
    local_update_acceptance: np.ndarray,
    J: float,
    h: float,
) -> None:
    R = states.shape[0]
    L = states.shape[1]
    n_steps = sites_by_walker.shape[1]
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
            s0 = int(states[walker, i, j])
            nn_sum = (
                int(states[walker, ip, j])
                + int(states[walker, im, j])
                + int(states[walker, i, jp])
                + int(states[walker, i, jm])
            )
            dE = 2.0 * s0 * (J * nn_sum + h)
            if dE <= 0.0 or accept_randoms[walker, step] < math.exp(-beta * dE):
                states[walker, i, j] = -s0
                energy_delta += dE
                accepted_count += 1
        energy_by_walker[walker] += energy_delta
        local_update_attempts[walker] += n_steps
        local_update_acceptance[walker] += accepted_count


@dataclass(frozen=True)
class IsingModel(BaseCPUModel):
    """
    Square-lattice Ising model for the simple CPU benchmark runner.
    """

    J: float = 1.0
    h: float = 0.0
    ordered_start: bool = False
    name: str = "ising_cpu"
    output_prefix: str = "ising2d_cpu"

    def prepare_states(self, states: list[np.ndarray]) -> np.ndarray:
        return np.ascontiguousarray(np.stack(states, axis=0), dtype=np.int8)

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
        del proposal_randoms
        _ising_sweep_walkers_numba(
            states,
            betas_by_walker,
            energy_by_walker,
            sites_by_walker,
            accept_randoms,
            local_update_attempts,
            local_update_acceptance,
            float(self.J),
            float(self.h),
        )

    def initial_state(
        self,
        L: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        self.validate_lattice(L)
        if self.ordered_start:
            return np.ones((int(L), int(L)), dtype=np.int8)
        return rng.choice(
            np.asarray([-1, 1], dtype=np.int8),
            size=(int(L), int(L)),
        )

    def energy(self, state: np.ndarray) -> float:
        spins = np.asarray(state)
        bond_sum = np.sum(
            spins * np.roll(spins, -1, axis=0),
            dtype=np.int64,
        )
        bond_sum += np.sum(
            spins * np.roll(spins, -1, axis=1),
            dtype=np.int64,
        )
        field_sum = np.sum(spins, dtype=np.int64)
        return float(-self.J * bond_sum - self.h * field_sum)

    def measure_observables(
        self,
        state: np.ndarray,
        beta: float,
    ) -> dict[str, float]:
        del beta
        M = float(np.sum(state, dtype=np.float64))
        M2 = M * M
        return {
            "order_parameter": M,
            "order_abs": abs(M),
            "order2": M2,
            "order4": M2 * M2,
        }

    def metadata(self) -> dict[str, object]:
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
            "update_scheme": "CPU random single-site Metropolis spin flips",
            "exact_Tc_h0": float(exact_Tc),
            "derived_observables": ["z2_magnetization"],
            "analysis_capabilities": {
                "thermodynamics": True,
                "z2_order_parameter": True,
                "helicity_modulus": False,
            },
        }


__all__ = ["IsingModel"]
