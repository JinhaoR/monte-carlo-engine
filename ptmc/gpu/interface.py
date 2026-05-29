from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ModelLaunchGeometry:
    """
    Launch sizes that depend on the model's update coloring.

    PTMC does not know whether a model uses two colors, four colors, or a
    different local update partition. The model reports how many update sites
    one walker touches for one color/pass, and the runner turns that into
    blocks and RNG state counts.
    """

    update_sites_per_walker: int
    update_blocks_per_walker: int
    update_rng_states: int


class BaseRuntime(ABC):
    """
    Base class for a model runtime used by the parallel tempering runner.

    The runtime owns the live model state like parameters, GPU arrays, energies, and observables.
    """

    @property
    @abstractmethod
    def energy_by_walker(self) -> Any:
        """
        Current energy of each walker.
        """
        raise NotImplementedError

    @abstractmethod
    def sweep(
        self,
        betas_by_walker: Any,
        rng_states_updates: Any,
        slot_of_walker: Any,
    ) -> None:
        """
        Perform one model Monte Carlo update sweep.

        This is where the model's Monte Carlo update rule lives.
        """
        raise NotImplementedError

    @abstractmethod
    def maybe_recompute_energy(
        self,
        sweeps_completed: int,
        recompute_stride: int,
        tolerance_per_site: np.float32,
    ) -> None:
        """
        Optionally recompute energy exactly to control numerical drift.
        """
        raise NotImplementedError

    @abstractmethod
    def allocate_measurement_storage(
        self,
        n_meas: int,
        n_derived_meas: int,
        store_primary_histories: bool,
        observable_n_blocks: int,
    ) -> None:
        """
        Allocate model specific measurement arrays.
        """
        raise NotImplementedError

    @abstractmethod
    def record_primary_observables(
        self,
        walker_of_slot: Any,
        col: int,
    ) -> None:
        """
        Record frequently measured observables.
        """
        raise NotImplementedError

    @abstractmethod
    def record_derived_observables(
        self,
        betas_by_walker: Any,
        walker_of_slot: Any,
        col: int,
    ) -> None:
        """
        Record more expensive observables.
        """
        raise NotImplementedError

    @abstractmethod
    def copy_measurements_to_host(self) -> dict[str, np.ndarray]:
        """
        Copy measurements from GPU memory to CPU NumPy arrays.
        """
        raise NotImplementedError

    @abstractmethod
    def sync_energy_drift_stats_from_gpu(self) -> dict[str, np.ndarray]:
        """
        Copy energy drift diagnostic arrays back to the CPU.
        """
        raise NotImplementedError
    
class BaseModel(ABC):
    """
    Base class for a model that can be used by the GPU parallel tempering runner.

    The model stores parameters and knows how to create its runtime.
    """

    @abstractmethod
    def max_threads_per_block(self) -> int:
        """
        Maximum CUDA block size supported by this model's kernels.
        """
        raise NotImplementedError

    def validate_lattice(self, L: int) -> None:
        """
        Validate model-specific lattice constraints.

        Models that use checkerboard updates can require even L here. PTMC only
        checks generic simulation constraints.
        """
        L = int(L)
        if L <= 0:
            raise ValueError("L must be positive.")

    @abstractmethod
    def update_sites_per_walker(self, L: int) -> int:
        """
        Number of update sites per walker for one model update color/pass.

        A two-color square-lattice checkerboard model usually returns
        L * (L // 2). A four-color model usually returns (L // 2) * (L // 2).
        """
        raise NotImplementedError

    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        """
        Return model information and parameters for saved output files.
        """
        raise NotImplementedError

    @abstractmethod
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
    ) -> BaseRuntime:
        """
        Create the live runtime for this model.
        """
        raise NotImplementedError

    def launch_geometry(
        self,
        *,
        L: int,
        R: int,
        threads_per_block: int,
    ) -> ModelLaunchGeometry:
        """
        Compute model-owned update launch geometry.
        """
        self.validate_lattice(L)
        update_sites_per_walker = int(self.update_sites_per_walker(L))
        if update_sites_per_walker <= 0:
            raise ValueError(
                "Model update_sites_per_walker(L) must return a positive integer."
            )
        threads_per_block = int(threads_per_block)
        update_blocks_per_walker = (
            update_sites_per_walker + threads_per_block - 1
        ) // threads_per_block
        return ModelLaunchGeometry(
            update_sites_per_walker=update_sites_per_walker,
            update_blocks_per_walker=update_blocks_per_walker,
            update_rng_states=int(R) * update_sites_per_walker,
        )


BasePTModel = BaseModel


def validate_pt_model(model: Any) -> BaseModel:
    """
    Validate the minimal model contract used by PTMC.
    """
    if not isinstance(model, BaseModel):
        missing = [
            name
            for name in (
                "max_threads_per_block",
                "metadata",
                "validate_lattice",
                "update_sites_per_walker",
                "launch_geometry",
                "create_runtime",
            )
            if not callable(getattr(model, name, None))
        ]
        if missing:
            raise TypeError(
                "model must implement the PTMC BaseModel contract. "
                f"Missing: {', '.join(missing)}."
            )
    return model
