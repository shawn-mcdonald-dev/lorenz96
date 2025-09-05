#!/usr/bin/env python3
# save as make_l96_dataset.py
import json, csv, math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

from l96lib import make_initial_conditions, simulate_l96

def one_sample(outdir: Path, sample_id: int, N: int, tmax: float, steps: int,
               F: float, ic_mode: str, eps: float, seed: int,
               method: str, rtol: float, atol: float):
    """
    Creates one sample: inputs_{id}.json, x0_{id}.npy, X_{id}.npy
    Returns a dict row for the manifest.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    x0 = make_initial_conditions(N, F, ic_mode, eps, seed, ic_file=None)
    t, X = simulate_l96(N, F, tmax, steps, x0, method, rtol, atol)
    # store float32 to save disk
    x0_path = outdir / f"x0_{sample_id:07d}.npy"
    X_path = outdir / f"X_{sample_id:07d}.npy"
    t_path = outdir / f"t_{sample_id:07d}.npy"
    np.save(x0_path, x0.astype(np.float64, copy=False))
    np.save(X_path, X.astype(np.float32, copy=False))
    np.save(t_path, t.astype(np.float32, copy=False))

    inputs = {
        "sample_id": sample_id,
        "N": N, "F": float(F),
        "tmax": float(tmax), "steps": int(steps),
        "method": method, "rtol": float(rtol), "atol": float(atol),
        "ic": {"mode": ic_mode, "eps": float(eps), "seed": int(seed),
               "x0_path": str(x0_path.name), "source": "generated"},
        "outputs": {"X_path": str(X_path.name), "t_path": str(t_path.name)}
    }
    inputs_path = outdir / f"inputs_{sample_id:07d}.json"
    with open(inputs_path, "w") as f:
        json.dump(inputs, f, indent=2)

    return {
        "sample_id": sample_id,
        "inputs_json": inputs_path.name,
        "x0_npy": x0_path.name,
        "X_npy": X_path.name,
        "t_npy": t_path.name,
        "F": F
    }

def main():
    import argparse, random
    p = argparse.ArgumentParser("Generate Lorenz-96 dataset")
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--num-samples", type=int, default=1000)
    p.add_argument("--N", type=int, default=40)
    p.add_argument("--tmax", type=float, default=20.0)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--F-min", type=float, default=4.0)
    p.add_argument("--F-max", type=float, default=16.0)
    p.add_argument("--ic-mode", choices=["steady", "random"], default="random")
    p.add_argument("--eps", type=float, default=2.0)
    p.add_argument("--method", type=str, default="RK4FIXED",
                   help='Integrator: "RK4FIXED" (deterministic) or scipy methods like "RK45"')
    p.add_argument("--rtol", type=float, default=1e-6)
    p.add_argument("--atol", type=float, default=1e-9)
    p.add_argument("--seed", type=int, default=12345, help="global seed for reproducibility")
    p.add_argument("--jobs", type=int, default=4)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.outdir / "manifest.csv"

    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = []
        for i in range(args.num_samples):
            # sample F and per-sample seed deterministically from global
            F = rng.uniform(args.F_min, args.F_max)
            seed_i = int(rng.integers(0, 2**31 - 1))
            futs.append(ex.submit(
                one_sample, args.outdir, i, args.N, args.tmax, args.steps,
                float(F), args.ic_mode, args.eps, seed_i,
                args.method, args.rtol, args.atol
            ))
        for fut in as_completed(futs):
            rows.append(fut.result())

    # write manifest
    rows.sort(key=lambda r: r["sample_id"])
    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id","inputs_json","x0_npy","X_npy","t_npy","F"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} samples and manifest: {manifest_path}")

if __name__ == "__main__":
    main()

