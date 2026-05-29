from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from ptmc.cpu.interface import BaseCPUModel


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
        old = float(state[i, j])
        new = old + rng.uniform(-self.theta_step, self.theta_step)

        old_sum = 0.0
        new_sum = 0.0
        for ni, nj in (
            ((i + 1) % L, j),
            ((i - 1) % L, j),
            (i, (j + 1) % L),
            (i, (j - 1) % L),
        ):
            neighbor = float(state[ni, nj])
            old_sum += math.cos(old - neighbor)
            new_sum += math.cos(new - neighbor)
        dE = self.J * (old_sum - new_sum)
        accepted = dE <= 0.0 or rng.random() < math.exp(-float(beta) * dE)
        if accepted:
            state[i, j] = new % (2.0 * math.pi)
            return float(dE), True
        return 0.0, False

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

    def metadata(self) -> dict[str, Any]:
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
