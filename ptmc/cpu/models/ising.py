from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ptmc.cpu.interface import BaseCPUModel


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
        spins = np.asarray(state, dtype=np.float64)
        bond_sum = np.sum(spins * np.roll(spins, -1, axis=0))
        bond_sum += np.sum(spins * np.roll(spins, -1, axis=1))
        field_sum = np.sum(spins)
        return float(-self.J * bond_sum - self.h * field_sum)

    def metropolis_step(
        self,
        state: np.ndarray,
        site: int,
        beta: float,
        rng: np.random.Generator,
    ) -> tuple[float, bool]:
        L = state.shape[0]
        i = int(site) // L
        j = int(site) - i * L
        s0 = int(state[i, j])
        nn_sum = int(
            state[(i + 1) % L, j]
            + state[(i - 1) % L, j]
            + state[i, (j + 1) % L]
            + state[i, (j - 1) % L]
        )
        dE = 2.0 * s0 * (self.J * nn_sum + self.h)
        accepted = dE <= 0.0 or rng.random() < np.exp(-float(beta) * dE)
        if accepted:
            state[i, j] = np.int8(-s0)
            return float(dE), True
        return 0.0, False

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
