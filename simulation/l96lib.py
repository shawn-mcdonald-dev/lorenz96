# l96lib.py
from __future__ import annotations
import numpy as np
from typing import Tuple, Optional
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def lorenz96(t, x, F: float):
    """Lorenz-96 ODE: dx_i/dt = (x_{i+1}-x_{i-2}) x_{i-1} - x_i + F"""
    N = len(x)
    d = np.empty(N, dtype=float)
    for i in range(N):
        xm2 = x[(i - 2) % N]
        xm1 = x[(i - 1) % N]
        xp1 = x[(i + 1) % N]
        d[i] = (xp1 - xm2) * xm1 - x[i] + F
    return d

def make_initial_conditions(
    N: int,
    F: float,
    ic_mode: str = "steady",
    eps: float = 0.01,
    seed: Optional[int] = None,
    ic_file: Optional[str] = None,
) -> np.ndarray:
    """Create or load an initial condition vector x0 of shape (N,)."""
    if ic_file is not None:
        x0 = np.load(ic_file)
        if x0.shape != (N,):
            raise ValueError(f"--ic-file shape {x0.shape} does not match N={N}")
        return x0.astype(float, copy=False)
    rng = np.random.default_rng(seed)
    if ic_mode == "steady":
        x0 = np.full(N, F, dtype=float)
        if eps > 0:
            x0 += rng.normal(0.0, eps, size=N)
    elif ic_mode == "random":
        x0 = rng.normal(F, eps if eps > 0 else 1.0, size=N)
    else:
        raise ValueError("ic_mode must be 'steady' or 'random'")
    return x0


def _rk4_fixed(l96_fun, x0: np.ndarray, F: float, tmax: float, steps: int) -> np.ndarray:
    """Deterministic fixed-step RK4 using dt = tmax/(steps-1)."""
    F = float(F)
    N = x0.size
    X = np.empty((N, steps), dtype=np.float64)
    x = x0.astype(np.float64, copy=True)
    X[:, 0] = x
    dt = float(tmax) / float(steps - 1)
    for k in range(steps - 1):
        k1 = l96_fun(0.0, x, F)
        k2 = l96_fun(0.0, x + 0.5 * dt * k1, F)
        k3 = l96_fun(0.0, x + 0.5 * dt * k2, F)
        k4 = l96_fun(0.0, x + dt * k3, F)
        x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        X[:, k + 1] = x
    return X


def simulate_l96(
    N: int, F: float, tmax: float, steps: int,
    x0: np.ndarray, method: str = "RK45",
    rtol: float = 1e-6, atol: float = 1e-9,
    t_eval: np.ndarray | None = None, 
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Integrate Lorenz-96 and return (t, X) where X has shape (N, steps).

    method:
      - "RK4FIXED": deterministic fixed-step RK4 using the provided t_eval
                    or a uniform grid from tmax/steps.
      - else: uses scipy.integrate.solve_ivp with t_eval.
    """
    if t_eval is None:
        t_eval = np.linspace(0.0, tmax, steps, dtype=float)
    else:
        # trust caller's grid; also keep steps in sync
        steps = int(t_eval.size)
        tmax = float(t_eval[-1])

    if method.upper() == "RK4FIXED":
        X = _rk4_fixed(lorenz96, x0, F, tmax, steps)
        return t_eval, X

    sol = solve_ivp(
        lorenz96, (float(t_eval[0]), float(t_eval[-1])),
        y0=x0, args=(F,), t_eval=t_eval,
        method=method, rtol=rtol, atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")
    return t_eval, sol.y


def plot_spacetime(
    t: np.ndarray, X: np.ndarray, title: str = "",
    cmap: str = "RdBu_r", vmin: Optional[float] = None, vmax: Optional[float] = None
):
    """Plot space-time heatmap (states x time). Returns a matplotlib Figure."""
    fig = plt.figure(figsize=(9, 5))
    plt.imshow(
        X, aspect="auto", origin="lower",
        extent=[t[0], t[-1], 0, X.shape[0]],
        cmap=cmap, vmin=vmin, vmax=vmax,
    )
    plt.xlabel("Time")
    plt.ylabel("State index (i)")
    if title:
        plt.title(title)
    cbar = plt.colorbar()
    cbar.set_label("x_i(t)")
    plt.tight_layout()
    return fig

def summarize_vector(x: np.ndarray, max_items: int = 12) -> str:
    """Concise printout of a long vector with simple stats."""
    n = x.size
    if n <= max_items:
        body = ", ".join(f"{v:.6g}" for v in x)
    else:
        head = ", ".join(f"{v:.6g}" for v in x[: max_items // 2])
        tail = ", ".join(f"{v:.6g}" for v in x[-(max_items // 2) :])
        body = f"{head}, …, {tail}"
    stats = dict(
        min=float(np.min(x)), max=float(np.max(x)),
        mean=float(np.mean(x)), std=float(np.std(x)),
        l2=float(np.linalg.norm(x)),
    )
    stats_str = ", ".join(f"{k}={v:.6g}" for k, v in stats.items())
    return f"[{body}]  (N={n}; {stats_str})"

