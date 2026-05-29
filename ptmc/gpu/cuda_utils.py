from __future__ import annotations

from numba import cuda


@cuda.jit
def fill_vector_kernel(a, value):
    idx = cuda.grid(1)
    if idx < a.shape[0]:
        a[idx] = value


@cuda.jit
def fill_two_vectors_kernel(a, b, value):
    idx = cuda.grid(1)
    if idx < a.shape[0]:
        a[idx] = value
        b[idx] = value


@cuda.jit
def fill_four_vectors_kernel(a, b, c, d, value):
    idx = cuda.grid(1)
    if idx < a.shape[0]:
        a[idx] = value
        b[idx] = value
        c[idx] = value
        d[idx] = value
