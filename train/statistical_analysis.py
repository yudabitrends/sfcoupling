"""General-purpose statistical helpers for multi-seed experiments."""
import math
from typing import Dict, List, Tuple

import numpy as np

T_TABLE = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    20: 2.086, 25: 2.060, 30: 2.042,
}


def _t_crit(df: int) -> float:
    if df <= 0:
        return float("nan")
    if df in T_TABLE:
        return T_TABLE[df]
    if df < 30:
        lo = max(k for k in T_TABLE if k <= df)
        hi = min(k for k in T_TABLE if k >= df)
        if lo == hi:
            return T_TABLE[lo]
        frac = (df - lo) / (hi - lo)
        return T_TABLE[lo] + frac * (T_TABLE[hi] - T_TABLE[lo])
    if df < 60:
        return 2.000
    return 1.960


def t_interval_ci(values, confidence: float = 0.95) -> Dict:
    """Return mean and 95% CI using t-interval."""
    arr = np.asarray(values, dtype=np.float64)
    n = arr.size
    mean = float(np.mean(arr))
    if n <= 1:
        return {"mean": mean, "ci_low": float("nan"), "ci_high": float("nan")}
    std = float(np.std(arr, ddof=1))
    t = _t_crit(n - 1)
    half = t * std / math.sqrt(n)
    return {"mean": mean, "ci_low": mean - half, "ci_high": mean + half}


def bootstrap_bca_ci(
    data: np.ndarray,
    n_boot: int = 10000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict:
    """BCa bootstrap confidence interval on the mean of rows.

    Parameters
    ----------
    data : array-like, shape (n_seeds,) or (n_seeds, n_features)
        Per-seed scalar or vector values.
    n_boot : int
        Number of bootstrap resamples.
    seed : int
        Random seed for reproducibility.
    alpha : float
        Significance level (default 0.05 for 95% CI).

    Returns
    -------
    dict with keys: mean, ci_low, ci_high
    """
    from scipy.stats import norm as _norm

    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    n = arr.shape[0]
    if n <= 1:
        mean = float(np.mean(arr))
        return {"mean": mean, "ci_low": float("nan"), "ci_high": float("nan")}

    rng = np.random.RandomState(seed)
    theta_hat = np.mean(arr, axis=0)  # (d,)

    # Bootstrap resamples
    boot_means = np.empty((n_boot, arr.shape[1]))
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boot_means[b] = np.mean(arr[idx], axis=0)

    # Aggregate to scalar (mean across features)
    theta_scalar = float(np.mean(theta_hat))
    boot_scalar = np.mean(boot_means, axis=1)  # (n_boot,)

    # Bias correction
    z0 = _norm.ppf(np.mean(boot_scalar < theta_scalar))

    # Acceleration via jackknife
    jack = np.empty(n)
    for i in range(n):
        jack[i] = np.mean(np.delete(arr, i, axis=0))
    jack_mean = np.mean(jack)
    num = np.sum((jack_mean - jack) ** 3)
    den = 6.0 * (np.sum((jack_mean - jack) ** 2) ** 1.5)
    a_hat = num / den if den != 0 else 0.0

    # Adjusted percentiles
    z_alpha = _norm.ppf(alpha / 2.0)
    z_1alpha = _norm.ppf(1.0 - alpha / 2.0)

    def _adj(z):
        return _norm.cdf(z0 + (z0 + z) / (1.0 - a_hat * (z0 + z)))

    p_low = max(0.0, min(1.0, _adj(z_alpha)))
    p_high = max(0.0, min(1.0, _adj(z_1alpha)))

    ci_low = float(np.percentile(boot_scalar, 100.0 * p_low))
    ci_high = float(np.percentile(boot_scalar, 100.0 * p_high))

    return {"mean": theta_scalar, "ci_low": ci_low, "ci_high": ci_high}


def paired_t_test(a: np.ndarray, b: np.ndarray) -> Dict:
    """Two-sided paired t-test between two arrays of per-seed values.

    Returns dict with keys: t_stat, p_value, mean_diff, ci_low, ci_high.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff = a - b
    n = diff.size
    mean_diff = float(np.mean(diff))
    if n <= 1:
        return {"t_stat": float("nan"), "p_value": float("nan"),
                "mean_diff": mean_diff, "ci_low": float("nan"), "ci_high": float("nan")}
    std_diff = float(np.std(diff, ddof=1))
    se = std_diff / math.sqrt(n)
    t_stat = mean_diff / se if se > 0 else float("inf")
    t_crit = _t_crit(n - 1)
    half = t_crit * se

    # Two-sided p-value approximation via t-distribution
    try:
        from scipy.stats import t as _tdist
        p_value = float(2.0 * _tdist.sf(abs(t_stat), df=n - 1))
    except ImportError:
        p_value = float("nan")

    return {
        "t_stat": float(t_stat),
        "p_value": p_value,
        "mean_diff": mean_diff,
        "ci_low": mean_diff - half,
        "ci_high": mean_diff + half,
    }


def run_paired_comparisons(
    conditions: Dict[str, np.ndarray],
    pairs: List[Tuple[str, str]],
) -> Dict:
    """Run paired t-tests for each (a, b) pair.

    Parameters
    ----------
    conditions : dict mapping method name -> array of per-seed scalar values
    pairs : list of (method_a, method_b) tuples

    Returns
    -------
    dict mapping "method_a_vs_method_b" -> paired_t_test result
    """
    results = {}
    for a_name, b_name in pairs:
        key = f"{a_name}_vs_{b_name}"
        results[key] = paired_t_test(conditions[a_name], conditions[b_name])
    return results
