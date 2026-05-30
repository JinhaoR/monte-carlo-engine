from __future__ import annotations

import numpy as np


def pack_two_color_checkerboard(field: np.ndarray) -> np.ndarray:
    """
    Pack a full (R, L, L) field into two square-lattice checkerboard colors.

    This is an optional helper. Models with different layouts can ignore it.
    """
    if field.ndim != 3 or field.shape[1] != field.shape[2]:
        raise ValueError("Expected a field with shape (R, L, L).")
    R, L, _ = field.shape
    if L % 2 != 0:
        raise ValueError("Two-color checkerboard packing requires even L.")

    half = L // 2
    packed = np.empty((R, 2, L, half), dtype=field.dtype)
    for color in (0, 1):
        for i in range(L):
            packed[:, color, i, :] = field[:, i, ((i + color) & 1) :: 2]
    return packed


def pack_four_color_checkerboard(field: np.ndarray) -> np.ndarray:
    """
    Pack a full (R, L, L) field into four square-lattice checkerboard colors.

    The color is ``2 * (i % 2) + (j % 2)``. This layout supports update
    stencils that couple a site to both forward neighbors in the same local
    energy term.
    """
    if field.ndim != 3 or field.shape[1] != field.shape[2]:
        raise ValueError("Expected a field with shape (R, L, L).")
    R, L, _ = field.shape
    if L % 2 != 0:
        raise ValueError("Four-color checkerboard packing requires even L.")

    half = L // 2
    packed = np.empty((R, 4, half, half), dtype=field.dtype)
    for i_parity in (0, 1):
        for j_parity in (0, 1):
            color = 2 * i_parity + j_parity
            packed[:, color, :, :] = field[
                :,
                i_parity::2,
                j_parity::2,
            ]
    return packed
