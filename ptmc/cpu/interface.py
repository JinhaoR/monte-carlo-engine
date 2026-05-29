from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseCPUModel(ABC):
    """
    Minimal contract for the CPU benchmark runner.

    CPU models own their NumPy state representation and batched update sweep.
    The runner handles parallel tempering, bookkeeping, and measurement storage.
    """

    needs_proposal_randoms = False

    def validate_lattice(self, L: int) -> None:
        L = int(L)
        if L <= 0:
            raise ValueError("L must be positive.")

    def sweep_sites_per_walker(self, L: int) -> int:
        self.validate_lattice(L)
        return int(L) * int(L)

    def prepare_states(self, states: list[Any]) -> Any:
        return states

    @abstractmethod
    def sweep_walkers(
        self,
        *,
        states: Any,
        betas_by_walker: np.ndarray,
        energy_by_walker: np.ndarray,
        sites_by_walker: np.ndarray,
        accept_randoms: np.ndarray,
        proposal_randoms: np.ndarray | None,
        local_update_attempts: np.ndarray,
        local_update_acceptance: np.ndarray,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def initial_state(self, L: int, rng: np.random.Generator) -> Any:
        raise NotImplementedError

    @abstractmethod
    def energy(self, state: Any) -> float:
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
                "sweep_walkers",
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
