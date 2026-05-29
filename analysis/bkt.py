from __future__ import annotations

from typing import Any

import numpy as np


ArrayLike = Any


def weber_minnhagen_line(
    temps: ArrayLike,
    L: int,
    C: float,
) -> np.ndarray:
    """
    Weber-Minnhagen finite-size helicity reference line.

    Y_L(T) = (2T / pi) * [1 + 1 / (2 log L + C)]
    """
    temps = np.asarray(temps, dtype=np.float64)

    L = int(L)
    if L <= 1:
        raise ValueError("L must be greater than 1.")

    denominator = 2.0 * np.log(float(L)) + float(C)

    if not np.isfinite(denominator) or denominator <= 0.0:
        return np.full_like(temps, np.nan, dtype=np.float64)

    return (2.0 * temps / np.pi) * (1.0 + 1.0 / denominator)


def bare_bkt_line(temps: ArrayLike) -> np.ndarray:
    """
    Bare universal BKT jump line.

    Y = 2T / pi
    """
    temps = np.asarray(temps, dtype=np.float64)
    return 2.0 * temps / np.pi


def helicity_reference_line(
    temps: ArrayLike,
    *,
    L: int | None = None,
    weber_minnhagen_C: float | None = None,
) -> np.ndarray:
    """
    Return either the bare BKT line or the Weber-Minnhagen corrected line.
    """
    if (
        L is not None
        and weber_minnhagen_C is not None
        and np.isfinite(weber_minnhagen_C)
    ):
        return weber_minnhagen_line(
            temps,
            L=L,
            C=float(weber_minnhagen_C),
        )

    return bare_bkt_line(temps)


def helicity_reference_label(
    weber_minnhagen_C: float | None = None,
) -> str:
    """
    Label for plotting the helicity reference line.
    """
    if weber_minnhagen_C is not None and np.isfinite(weber_minnhagen_C):
        return r"$(2T/\pi)[1+1/(2\log L+C)]$"

    return r"$2T/\pi$"


def _piecewise_linear_root(
    x: np.ndarray,
    y: np.ndarray,
) -> float | None:
    """
    Find one root of y(x) by piecewise-linear interpolation.

    If there are multiple crossings, return the one with smallest local
    absolute mismatch.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]

    if x.size < 2:
        return None

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    candidates: list[tuple[float, float]] = []

    for i in range(x.size - 1):
        x0 = float(x[i])
        x1 = float(x[i + 1])
        y0 = float(y[i])
        y1 = float(y[i + 1])

        if y0 == 0.0:
            candidates.append((x0, abs(y0)))
            continue

        if y0 * y1 > 0.0:
            continue

        if y1 == y0:
            root = 0.5 * (x0 + x1)
        else:
            root = x0 - y0 * (x1 - x0) / (y1 - y0)

        if x0 <= root <= x1:
            candidates.append((float(root), min(abs(y0), abs(y1))))

    if y[-1] == 0.0:
        candidates.append((float(x[-1]), 0.0))

    if not candidates:
        return None

    return min(candidates, key=lambda item: item[1])[0]


def estimate_bkt_intersection(
    *,
    temps: ArrayLike,
    Y: ArrayLike,
    L: int,
    Y_err: ArrayLike | None = None,
    weber_minnhagen_C: float | None = None,
    n_bootstrap: int = 2000,
    rng_seed: int = 12345,
) -> dict[str, Any]:
    """
    Estimate the temperature where measured Y intersects the BKT reference line.

    If weber_minnhagen_C is None, use Y = 2T/pi.
    Otherwise use the Weber-Minnhagen finite-size line.
    """
    temps = np.asarray(temps, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    if temps.shape != Y.shape:
        raise ValueError("temps and Y must have the same shape.")

    reference = helicity_reference_line(
        temps,
        L=L,
        weber_minnhagen_C=weber_minnhagen_C,
    )

    diff = Y - reference
    T_cross = _piecewise_linear_root(temps, diff)

    if T_cross is None:
        return {
            "available": False,
            "L": int(L),
            "reason": "no sign-changing helicity/reference intersection",
            "weber_minnhagen_C": (
                np.nan
                if weber_minnhagen_C is None
                else float(weber_minnhagen_C)
            ),
        }

    T_err = np.nan
    n_used = 0

    if Y_err is not None and n_bootstrap > 1:
        Y_err = np.asarray(Y_err, dtype=np.float64)

        if Y_err.shape == Y.shape:
            rng = np.random.default_rng(rng_seed)
            roots = []

            for _ in range(int(n_bootstrap)):
                sample_Y = Y + rng.normal(0.0, np.maximum(Y_err, 0.0))
                sample_diff = sample_Y - reference
                root = _piecewise_linear_root(temps, sample_diff)

                if root is not None:
                    roots.append(root)

            roots = np.asarray(roots, dtype=np.float64)
            roots = roots[np.isfinite(roots)]

            n_used = int(roots.size)

            if roots.size >= 2:
                T_err = float(np.std(roots, ddof=1))

    return {
        "available": True,
        "L": int(L),
        "T": float(T_cross),
        "T_err": float(T_err),
        "line_label": helicity_reference_label(weber_minnhagen_C),
        "weber_minnhagen_C": (
            np.nan
            if weber_minnhagen_C is None
            else float(weber_minnhagen_C)
        ),
        "n_bootstrap": int(n_used),
    }


def estimate_bkt_intersections_by_L(
    analyzed_by_L: dict[int, dict[str, Any]],
    *,
    weber_minnhagen_C: float | None = None,
    n_bootstrap: int = 2000,
    rng_seed: int = 12345,
) -> dict[int, dict[str, Any]]:
    """
    Estimate BKT intersections for every L that has Y data.
    """
    estimates: dict[int, dict[str, Any]] = {}

    for offset, L in enumerate(sorted(analyzed_by_L)):
        obs = analyzed_by_L[L]

        if "temps" not in obs or "Y" not in obs:
            continue

        estimates[int(L)] = estimate_bkt_intersection(
            temps=obs["temps"],
            Y=obs["Y"],
            Y_err=obs.get("Y_err"),
            L=int(L),
            weber_minnhagen_C=weber_minnhagen_C,
            n_bootstrap=n_bootstrap,
            rng_seed=int(rng_seed) + 1009 * offset,
        )

    return estimates