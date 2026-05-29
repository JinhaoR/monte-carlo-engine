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
