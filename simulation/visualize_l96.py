#!/usr/bin/env python3
# visualize_l96.py
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def load_run(path: Path, sample_id: int | None):
    """
    Load (t, X, x0, meta, title_path) from either:
      - a dataset dir + --id (uses inputs_#######.json), or
      - a direct JSON path (inputs_#######.json), or
      - a folder containing X.npy, t.npy, x0.npy
    """
    p = Path(path)
    if p.is_file() and p.name.startswith("inputs_") and p.suffix == ".json":
        base = p.parent
        meta = json.loads(p.read_text())
        t = np.load(base / meta["outputs"]["t_path"])
        X = np.load(base / meta["outputs"]["X_path"])
        x0 = np.load(base / meta["ic"]["x0_path"])
        title = f"{p.name}"
        return t, X, x0, meta, title

    if p.is_dir() and sample_id is not None:
        j = p / f"inputs_{sample_id:07d}.json"
        if not j.exists():
            raise SystemExit(f"Could not find {j}")
        return load_run(j, None)

    if p.is_dir():
        # assume out/ style folder with X.npy, t.npy, x0.npy
        t = np.load(p / "t.npy")
        X = np.load(p / "X.npy")
        x0p = p / "x0.npy"
        x0 = np.load(x0p) if x0p.exists() else X[:, 0]
        meta = dict(N=int(X.shape[0]), steps=int(X.shape[1]), tmax=float(t[-1]))
        title = f"{p}/(X.npy, t.npy)"
        return t, X, x0, meta, title

    raise SystemExit(f"Unsupported input {path}")

def plot_spacetime(t, X, cmap="RdBu_r", vmin=None, vmax=None, title="Space–time (states × time)"):
    plt.figure(figsize=(10, 4.5))
    plt.imshow(X, aspect="auto", origin="lower",
               extent=[t[0], t[-1], 0, X.shape[0]],
               cmap=cmap, vmin=vmin, vmax=vmax)
    plt.xlabel("Time")
    plt.ylabel("State index i")
    plt.title(title)
    cbar = plt.colorbar()
    cbar.set_label("x_i(t)")
    plt.tight_layout()

def plot_traces(t, X, idx=(0, 10, 20, 30), title="Selected traces"):
    plt.figure(figsize=(10, 3.2))
    for i in idx:
        if 0 <= i < X.shape[0]:
            plt.plot(t, X[i, :], label=f"x[{i}]")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title(title + f"  (N={X.shape[0]})")
    plt.legend(ncol=4, fontsize=8, frameon=False)
    plt.tight_layout()

def plot_x0(x0, title="Initial condition x0"):
    plt.figure(figsize=(10, 2.8))
    plt.bar(np.arange(x0.size), x0, width=0.8)
    plt.xlabel("State index i")
    plt.ylabel("x0[i]")
    plt.title(title)
    plt.tight_layout()

def plot_wavenumber_spectrum(X, title="Mean wavenumber spectrum (time-avg)"):
    """
    Spatial spectrum over state index i.
    Compute FFT along axis 0 for each time, average |fft|^2 over time.
    """
    N, T = X.shape
    # remove time-mean at each i to reduce DC dominance
    Xc = X - X.mean(axis=1, keepdims=True)
    # FFT over i (states)
    Sk = np.fft.rfft(Xc, axis=0)
    Pk = (Sk * np.conj(Sk)).real  # power
    Pk_mean = Pk.mean(axis=1)     # average over time
    k = np.arange(Pk_mean.size)   # wavenumber bins
    plt.figure(figsize=(10, 3.0))
    plt.plot(k, Pk_mean)
    plt.xlabel("Wavenumber k")
    plt.ylabel("Power")
    plt.title(title + f"  (N={N})")
    plt.tight_layout()

def main():
    ap = argparse.ArgumentParser("Visualize a Lorenz-96 run (space–time image, traces, x0, spectrum).")
    ap.add_argument("--path", type=Path, required=True,
                    help="Dataset dir, out/ dir, or inputs_#######.json")
    ap.add_argument("--id", type=int, default=None, help="Sample id (if --path is a dataset dir)")
    ap.add_argument("--cmap", type=str, default="RdBu_r")
    ap.add_argument("--vmin", type=float, default=None)
    ap.add_argument("--vmax", type=float, default=None)
    ap.add_argument("--trace-idx", type=int, nargs="*", default=[0, 10, 20, 30],
                    help="Indices to plot in the time-series panel")
    ap.add_argument("--save-prefix", type=Path, default=None,
                    help="If set, save PNGs with this prefix instead of showing windows.")
    args = ap.parse_args()

    t, X, x0, meta, title_path = load_run(args.path, args.id)

    # Space-time heatmap
    plot_spacetime(t, X, cmap=args.cmap, vmin=args.vmin, vmax=args.vmax,
                   title=f"{title_path}  |  heatmap")

    # Time traces
    plot_traces(t, X, idx=args.trace_idx, title=f"{title_path}  |  traces")

    # Initial condition
    plot_x0(x0, title=f"{title_path}  |  x0")

    # Wavenumber spectrum
    plot_wavenumber_spectrum(X, title=f"{title_path}  |  spectrum")

    if args.save_prefix is not None:
        args.save_prefix.parent.mkdir(parents=True, exist_ok=True)
        for i, fig in enumerate(plt.get_fignums(), start=1):
            plt.figure(fig)
            out = f"{args.save_prefix}.{i:02d}.png"
            plt.savefig(out, dpi=180, bbox_inches="tight")
            print("Saved", out)
    else:
        plt.show()

if __name__ == "__main__":
    main()

