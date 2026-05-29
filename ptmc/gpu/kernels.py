from __future__ import annotations
import math
from numba import cuda, float32
from numba.cuda.random import xoroshiro128p_uniform_float32

@cuda.jit
def parallel_tempering_swap_kernel(
    energy_by_walker,
    betas,
    walker_of_slot,
    slot_of_walker,
    betas_by_walker,
    rng_states,
    parity,
    swap_attempts,
    swap_acceptance,
):
    """
    Attempt neighboring parallel tempering swaps on the GPU.

    This kernel swaps walkers between neighboring temperature slots.

    The model specific part is only energy_by_walker.
    Everything else is PT bookkeeping.
    """
    pair_idx = cuda.grid(1)
    R = walker_of_slot.shape[0]
    slot = parity + 2 * pair_idx
    if slot + 1 >= R:
        return
    wi = walker_of_slot[slot]
    wj = walker_of_slot[slot + 1]
    beta_i = betas[slot]
    beta_j = betas[slot + 1]
    energy_i = energy_by_walker[wi]
    energy_j = energy_by_walker[wj]
    delta = (beta_i - beta_j) * (energy_j - energy_i)
    swap_attempts[slot] += 1
    accept = delta <= float32(0.0)
    if not accept:
        u = xoroshiro128p_uniform_float32(rng_states, pair_idx)
        accept = u < float32(math.exp(-delta))
    if accept:
        walker_of_slot[slot] = wj
        walker_of_slot[slot + 1] = wi
        slot_of_walker[wi] = slot + 1
        slot_of_walker[wj] = slot
        betas_by_walker[wi] = beta_j
        betas_by_walker[wj] = beta_i
        swap_acceptance[slot] += 1

@cuda.jit
def record_positions_kernel(
    slot_of_walker,
    out,
    row,
):
    """
    Record the current temperature slot position of each walker.

    out[row, walker] = current slot of that walker
    """
    walker = cuda.grid(1)
    if walker < slot_of_walker.shape[0]:
        out[row, walker] = slot_of_walker[walker]