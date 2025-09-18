#!/usr/bin/env python3
# l96_metrics_cli.py
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
from l96_metrics import time_mean_energy

def _load_single_run(run_path: Path, sample_id: int | None):
    """
    Load (t, X, meta, run_id, src) from:
      - a dataset dir + --id (uses inputs_#######.json), or
      - an inputs_#######.json file directly, or
      - a folder containing X.npy, t.npy (and optional x0.npy).
    """
    p = run_path
    # Case A: direct JSON file
    if p.is_file() and p.suffix == ".json" and p.name.startswith("inputs_"):
        base = p.parent
        meta = json.loads(p.read_text())
        X = np.load(base / meta["outputs"]["X_path"])
        t = np.load(base / meta["outputs"]["t_path"])
        run_id = p.stem.split("_")[-1]
        return t, X, meta, run_id, str(p)

    # Case B: dataset dir + id
    if p.is_dir() and sample_id is not None:
        j = p / f"inputs_{sample_id:07d}.json"
        if not j.exists():
            raise SystemExit(f"Missing {j}")
        return _load_single_run(j, None)

    # Case C: simple folder with X.npy, t.npy
    if p.is_dir():
        Xp, tp = p / "X.npy", p / "t.npy"
        if not (Xp.exists() and tp.exists()):
            raise SystemExit(f"Could not find X.npy and t.npy under: {p}")
        X = np.load(Xp)
        t = np.load(tp)
        meta = {"path": str(p), "N": int(X.shape[0]), "steps": int(X.shape[1]), "tmax": float(t[-1])}
        run_id = p.name
        return t, X, meta, run_id, str(p)

    raise SystemExit(f"Unsupported input: {run_path}")

def _iter_dataset(dir_path: Path):
    """
    Yield (json_path, run_id) for every inputs_*.json in a dataset directory.
    """
    for j in sorted(dir_path.glob("inputs_*.json")):
        run_id = j.stem.split("_")[-1]
        yield j, run_id

def main():
    ap = argparse.ArgumentParser(
        description="Compute time-mean energy for Lorenz-96 runs and write a CSV."
    )
    ap.add_argument("--path", type=Path, required=True,
                    help="Dataset dir, an inputs_#######.json, or a folder with X.npy/t.npy.")
    ap.add_argument("--id", type=int, default=None,
                    help="Sample id if --path is a dataset directory.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Where to write CSV (default: <path>/targets_time_mean_energy.csv for dataset dir; stdout for single run).")
    ap.add_argument("--per-variable", action="store_true",
                    help="Report Ebar/N instead of total energy.")
    args = ap.parse_args()

    # Single-run mode if:
    #  - path is an inputs_*.json, OR
    #  - path is a dir with an explicit --id, OR
    #  - path is a dir containing X.npy and t.npy
    is_single_dir = args.path.is_dir() and (args.path / "X.npy").exists() and (args.path / "t.npy").exists()
    if args.path.is_file() or (args.path.is_dir() and (args.id is not None or is_single_dir)):
        t, X, meta, run_id, src = _load_single_run(args.path, args.id)
        Ebar = time_mean_energy(X, t)
        if args.per_variable:
            Ebar = Ebar / X.shape[0]
        print("id,N,steps,tmax,E_time_mean" + ("_per_var" if args.per_variable else ""))
        print(f"{run_id},{X.shape[0]},{X.shape[1]},{t[-1]:.6g},{Ebar:.8g}")
        return
    
    # Dataset directory (many runs)
    if not args.path.is_dir():
        raise SystemExit(f"--path must be a directory for dataset mode: {args.path}")

    out_csv = args.out or (args.path / ("targets_time_mean_energy" + ("_per_var" if args.per_variable else "") + ".csv"))
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for jpath, run_id in _iter_dataset(args.path):
        base = jpath.parent
        meta = json.loads(jpath.read_text())
        X = np.load(base / meta["outputs"]["X_path"])
        t = np.load(base / meta["outputs"]["t_path"])
        Ebar = time_mean_energy(X, t)
        if args.per_variable:
            Ebar = Ebar / X.shape[0]
        rows.append({
            "id": run_id,
            "N": int(meta["N"]),
            "steps": int(meta["steps"]),
            "tmax": float(meta["tmax"]),
            "F": float(meta["F"]),
            "E_time_mean" + ("_per_var" if args.per_variable else ""): Ebar,
        })

    # Write CSV
    fieldnames = list(rows[0].keys()) if rows else ["id","N","steps","tmax","F","E_time_mean"]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {len(rows)} rows to {out_csv}")

if __name__ == "__main__":
    main()

