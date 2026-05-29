from __future__ import annotations

from numba import cuda


def block_count_and_size(n_samples: int, requested_blocks: int) -> tuple[int, int]:
    """
    Choose contiguous block count and block size for measurement summaries.
    """
    requested_blocks = max(0, int(requested_blocks))
    n_samples = int(n_samples)
    if n_samples <= 0 or requested_blocks <= 0:
        return 0, 0
    n_blocks = min(requested_blocks, n_samples // 2)
    if n_blocks < 2:
        n_blocks = 1
    return n_blocks, n_samples // n_blocks


@cuda.jit
def record_scalar_by_slot_kernel(value_by_walker, walker_of_slot, out, col):
    slot = cuda.grid(1)
    if slot < walker_of_slot.shape[0]:
        walker = walker_of_slot[slot]
        out[slot, col] = value_by_walker[walker]


@cuda.jit
def accumulate_scalar_block_moments_by_slot_kernel(
    value_by_walker,
    walker_of_slot,
    value_block_sums,
    value2_block_sums,
    col,
    block_size,
):
    slot = cuda.grid(1)
    if slot >= walker_of_slot.shape[0] or block_size <= 0:
        return
    block = col // block_size
    if block >= value_block_sums.shape[1]:
        return

    walker = walker_of_slot[slot]
    value = value_by_walker[walker]
    value_block_sums[slot, block] += value
    value2_block_sums[slot, block] += value * value


@cuda.jit
def accumulate_order_block_moments_by_slot_kernel(
    order_by_walker,
    walker_of_slot,
    order_abs_block_sums,
    order2_block_sums,
    order4_block_sums,
    col,
    block_size,
):
    slot = cuda.grid(1)
    if slot >= walker_of_slot.shape[0] or block_size <= 0:
        return
    block = col // block_size
    if block >= order_abs_block_sums.shape[1]:
        return

    walker = walker_of_slot[slot]
    order = order_by_walker[walker]
    order_abs = order if order >= 0.0 else -order
    order2 = order * order
    order_abs_block_sums[slot, block] += order_abs
    order2_block_sums[slot, block] += order2
    order4_block_sums[slot, block] += order2 * order2
