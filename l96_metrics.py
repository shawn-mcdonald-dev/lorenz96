# l96_metrics.py
from __future__ import annotations
import numpy as np

def time_mean_energy(X: np.ndarray, t: np.ndarray) -> float:
    """
    Time-mean energy for Lorenz-96 trajectory.

    Parameters
    ----------
    X : (N, steps) array
        State trajectory x_i(t_j).
    t : (steps,) array
        Time grid (not necessarily float64; can be float32).

    Returns
    -------
    float
        Time-mean energy = (1 / (t[-1]-t[0])) * ∫ 0.5 * sum_i x_i(t)^2 dt
        (computed with trapezoid rule).
    """
    # promote to float64 for numerics, but do not modify inputs
    X64 = X.astype(np.float64, copy=False)
    t64 = t.astype(np.float64, copy=False)
    # energy(t_j) = 0.5 * sum_i x_i(t_j)^2
    E_t = 0.5 * np.sum(X64 * X64, axis=0)
    # mean over time via integral / duration
    duration = float(t64[-1] - t64[0])
    if duration <= 0:
        raise ValueError("t must be increasing with at least two points.")
    Ebar = float(np.trapz(E_t, t64) / duration)
    return Ebar

