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


def _finite_curve_for_fit(
    temps: ArrayLike,
    values: ArrayLike,
    errors: ArrayLike | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return finite temperature/value/error arrays for Weber-Minnhagen fitting.
    Non-finite or missing errors are kept as NaN and handled downstream.
    """
    temps = np.asarray(temps, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)

    if errors is None:
        errors_arr = np.full_like(values, np.nan, dtype=np.float64)
    else:
        errors_arr = np.asarray(errors, dtype=np.float64)
        if errors_arr.shape != values.shape:
            errors_arr = np.full_like(values, np.nan, dtype=np.float64)

    finite = np.isfinite(temps) & np.isfinite(values)
    return temps[finite], values[finite], errors_arr[finite]


def _sorted_unique_curve_for_fit(
    temps: np.ndarray,
    values: np.ndarray,
    errors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sort by temperature and keep one point per temperature.
    """
    order = np.argsort(temps)
    temps = np.asarray(temps, dtype=np.float64)[order]
    values = np.asarray(values, dtype=np.float64)[order]
    errors = np.asarray(errors, dtype=np.float64)[order]
    temps, unique_idx = np.unique(temps, return_index=True)
    return temps, values[unique_idx], errors[unique_idx]


def _fit_constant_with_errors(
    values: ArrayLike,
    errors: ArrayLike,
) -> tuple[float, float, float, int]:
    """
    Fit a constant to values with optional one-sigma errors.

    If usable positive errors are available, use inverse-variance weights.
    Otherwise use an unweighted mean.  If chi2/dof > 1, inflate the fitted
    error by sqrt(chi2/dof), which is a useful guard when the finite-size form
    is only approximately satisfied by the available sizes.
    """
    values = np.asarray(values, dtype=np.float64)
    errors = np.asarray(errors, dtype=np.float64)

    finite = np.isfinite(values)
    values = values[finite]
    errors = errors[finite]
    n = int(values.size)

    if n == 0:
        return np.nan, np.nan, np.nan, 0

    positive_errors = np.isfinite(errors) & (errors > 0.0)
    if np.any(positive_errors):
        floor = float(np.nanmedian(errors[positive_errors])) * 1.0e-6
        sigma = np.where(positive_errors, np.maximum(errors, floor), np.nan)
        missing = ~np.isfinite(sigma)
        if np.any(missing):
            sigma[missing] = float(np.nanmedian(sigma[~missing]))
        weights = 1.0 / (sigma * sigma)
        mean = float(np.sum(weights * values) / np.sum(weights))
        err = float(np.sqrt(1.0 / np.sum(weights)))
    else:
        weights = np.ones(n, dtype=np.float64)
        mean = float(np.mean(values))
        err = float(np.std(values, ddof=1) / np.sqrt(n)) if n > 1 else 0.0

    chi2 = float(np.sum(weights * (values - mean) ** 2))
    dof = max(n - 1, 0)
    if dof > 0 and np.isfinite(err):
        red_chi2 = chi2 / dof
        if red_chi2 > 1.0:
            err *= float(np.sqrt(red_chi2))

    return mean, err, chi2, dof


def estimate_weber_minnhagen_C(
    analyzed_by_L: dict[int, dict[str, Any]],
    *,
    min_sizes: int = 3,
    scan_lo: float | None = None,
    scan_hi: float | None = None,
    scan_points: int = 5000,
    require_all_sizes: bool = True,
) -> dict[str, Any]:
    """
    Estimate the Weber-Minnhagen constant C from multiple helicity curves.

    The Weber-Minnhagen finite-size relation at T_BKT is

        Y_L(T) = (2T/pi) * [1 + 1 / (2 log L + C)].

    Rearranged at a candidate transition temperature T,

        C_L(T) = [pi Y_L(T)/(2T) - 1]^(-1) - 2 log L.

    This scans the shared temperature interval and chooses the temperature
    where the inferred C_L values are most consistent across system sizes.
    The returned C is the weighted constant fit to the C_L values at that
    selected temperature.
    """
    min_sizes = max(2, int(min_sizes))
    scan_points = max(2, int(scan_points))

    curves: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    lows: list[float] = []
    highs: list[float] = []

    for L in sorted(int(key) for key in analyzed_by_L):
        obs = analyzed_by_L[L]
        if "temps" not in obs or "Y" not in obs:
            continue

        temps, values, errors = _finite_curve_for_fit(
            obs["temps"],
            obs["Y"],
            obs.get("Y_err"),
        )
        if temps.size < 2:
            continue

        temps, values, errors = _sorted_unique_curve_for_fit(
            temps,
            values,
            errors,
        )
        if temps.size < 2:
            continue

        curves[int(L)] = (temps, values, errors)
        lows.append(float(np.min(temps)))
        highs.append(float(np.max(temps)))

    if len(curves) < min_sizes:
        return {
            "available": False,
            "reason": f"need at least {min_sizes} sizes with helicity data",
            "n_sizes": int(len(curves)),
        }

    lo = max(lows)
    hi = min(highs)

    if scan_lo is not None and np.isfinite(scan_lo):
        lo = max(lo, float(scan_lo))
    if scan_hi is not None and np.isfinite(scan_hi):
        hi = min(hi, float(scan_hi))

    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        return {
            "available": False,
            "reason": "no shared temperature interval across sizes after scan limits",
            "n_sizes": int(len(curves)),
        }

    candidate_temps = np.linspace(lo, hi, scan_points, dtype=np.float64)
    rows: list[dict[str, Any]] = []

    for T in candidate_temps:
        if T <= 0.0 or not np.isfinite(T):
            continue

        bare_line_value = 2.0 * float(T) / np.pi
        c_values: list[float] = []
        c_errors: list[float] = []
        y_values: list[float] = []
        used_sizes: list[int] = []

        for L, (temps, values, errors) in curves.items():
            Y = float(np.interp(T, temps, values))
            Y_err = float(np.interp(T, temps, errors))

            delta = Y / bare_line_value - 1.0
            if not np.isfinite(delta) or delta <= 0.0:
                continue

            C_L = 1.0 / delta - 2.0 * np.log(float(L))

            delta_err = Y_err / bare_line_value
            C_err = (
                delta_err / (delta * delta)
                if np.isfinite(delta_err) and delta_err > 0.0
                else np.nan
            )

            c_values.append(float(C_L))
            c_errors.append(float(C_err))
            y_values.append(float(Y))
            used_sizes.append(int(L))

        if require_all_sizes and len(c_values) != len(curves):
            continue
        if len(c_values) < min_sizes:
            continue

        C, C_err, chi2, dof = _fit_constant_with_errors(c_values, c_errors)
        red_chi2 = chi2 / dof if dof > 0 else np.nan

        rows.append(
            {
                "T": float(T),
                "C": float(C),
                "C_err": float(C_err),
                "C_values": np.asarray(c_values, dtype=np.float64),
                "C_errors": np.asarray(c_errors, dtype=np.float64),
                "Y_values": np.asarray(y_values, dtype=np.float64),
                "sizes": used_sizes,
                "chi2": float(chi2),
                "dof": int(dof),
                "red_chi2": float(red_chi2),
            }
        )

    if not rows:
        return {
            "available": False,
            "reason": "no scan temperature had enough sizes above the bare BKT line",
            "scan_lo": float(lo),
            "scan_hi": float(hi),
            "scan_points": int(scan_points),
            "n_sizes": int(len(curves)),
        }

    def row_score(row: dict[str, Any]) -> tuple[float, float]:
        red_chi2 = float(row.get("red_chi2", np.nan))
        if not np.isfinite(red_chi2):
            red_chi2 = np.inf
        return red_chi2, abs(float(row["T"]) - 0.5 * (lo + hi))

    ordered = sorted(rows, key=row_score)
    best = dict(ordered[0])

    best["available"] = True
    best["scan_lo"] = float(lo)
    best["scan_hi"] = float(hi)
    best["scan_points"] = int(scan_points)
    best["require_all_sizes"] = bool(require_all_sizes)
    best["n_sizes"] = int(len(best.get("sizes", [])))
    best["top_candidates"] = ordered[: min(5, len(ordered))]
    best["convention"] = "Y_L=(2T/pi)*(1+1/(2 log L + C))"

    return best

