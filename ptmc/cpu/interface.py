from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseCPUModel(ABC):
    """
    Minimal contract for the CPU benchmark runner.

    CPU models own their NumPy state and one single-site Metropolis update.
    The runner only handles parallel tempering, bookkeeping, and measurement
    storage.
    """

    def validate_lattice(self, L: int) -> None:
        L = int(L)
        if L <= 0:
            raise ValueError("L must be positive.")

    def sweep_sites_per_walker(self, L: int) -> int:
        self.validate_lattice(L)
        return int(L) * int(L)

    @abstractmethod
    def initial_state(self, L: int, rng: np.random.Generator) -> Any:
        raise NotImplementedError

    @abstractmethod
    def energy(self, state: Any) -> float:
        raise NotImplementedError

    @abstractmethod
    def metropolis_step(
        self,
        state: Any,
        site: int,
        beta: float,
        rng: np.random.Generator,
    ) -> tuple[float, bool]:
        """
        Mutate one site if accepted and return (delta_energy, accepted).
        """
        raise NotImplementedError

    @abstractmethod
    def measure_observables(
        self,
        state: Any,
        beta: float,
    ) -> dict[str, float]:
        """
        Return scalar observables for one walker.

        Keys are saved as '<key>_block_means'. Conventional keys include:
        order_abs, order2, order4, helicity_Kx, helicity_Ix, helicity_Ix2.
        """
        raise NotImplementedError

    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        raise NotImplementedError


def validate_cpu_model(model: Any) -> BaseCPUModel:
    if not isinstance(model, BaseCPUModel):
        missing = [
            name
            for name in (
                "validate_lattice",
                "sweep_sites_per_walker",
                "initial_state",
                "energy",
                "metropolis_step",
                "measure_observables",
                "metadata",
            )
            if not callable(getattr(model, name, None))
        ]
        if missing:
            raise TypeError(
                "model must implement the CPU PTMC model contract. "
                f"Missing: {', '.join(missing)}."
            )
    return model
