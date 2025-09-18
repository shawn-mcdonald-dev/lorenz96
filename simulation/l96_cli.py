#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import sys
from l96lib import (
    make_initial_conditions, simulate_l96, plot_spacetime, summarize_vector
)

def _resolve_inputs_path(replay_path: Path, replay_id: int | None) -> Path:
    if replay_path.is_file():
        return replay_path
    if replay_path.is_dir():
        if replay_id is None:
            raise SystemExit("When --replay points to a directory, you must pass --replay-id.")
        return replay_path / f"inputs_{replay_id:07d}.json"
    raise SystemExit(f"--replay path not found: {replay_path}")

def _compare_arrays(name, A, B, strict: bool, rtol: float, atol: float) -> tuple[bool, float]:
    if strict:
        ok = np.array_equal(A, B)
    else:
        ok = np.allclose(A, B, rtol=rtol, atol=atol)
    max_err = float(np.max(np.abs(A - B)))
    print(f"{name}: {'OK' if ok else 'MISMATCH'}  "
          f"{'array_equal' if strict else f'allclose(rtol={rtol}, atol={atol})'};  "
          f"max|Δ|={max_err:.3e}")
    return ok, max_err

def replay(replay_path: Path, replay_id: int | None,
           save_dir: Path | None, save_x0: Path | None,
           force_method: str | None, strict: bool,
           rtol: float, atol: float) -> int:
    """
    Replay a saved sample and compare outputs.
    Supports two JSON schemas:
      (A) dataset 'inputs_#######.json' with:
          meta['ic']['x0_path'], meta['outputs']['X_path'], meta['outputs']['t_path']
      (B) config 'run.json' with:
          meta['outputs']['x0'] (optional), and X/t saved as X.npy, t.npy in same dir.
    Returns: 0 if match, 1 otherwise.
    """
    jpath = _resolve_inputs_path(replay_path, replay_id)
    meta = json.loads(jpath.read_text())
    base = jpath.parent

    # --- Pull dynamics (works for both schemas) ---
    N      = int(meta["N"])
    F      = float(meta["F"])
    tmax   = float(meta["tmax"])
    steps  = int(meta["steps"])
    method = meta.get("method", "RK45")
    if force_method is not None:
        method = force_method
    rtol_cfg = float(meta.get("rtol", 1e-6))
    atol_cfg = float(meta.get("atol", 1e-9))
    use_rtol = rtol if force_method is not None else rtol_cfg
    use_atol = atol if force_method is not None else atol_cfg

    # --- Locate files depending on schema ---
    # x0
    def _resolve_x0_from_config(meta_outputs: dict, base: Path) -> Path:
        # meta_outputs["x0"] might be "out/x0.npy" (CWD-relative) or just "x0.npy"
        raw = Path(meta_outputs["x0"])
        if raw.is_absolute():
            return raw
        # Try (1) base/relative, (2) relative-as-given, (3) base/filename
        candidates = [base / raw, raw, base / raw.name]
        for c in candidates:
            if c.exists():
                return c
        # If none exist, return the first candidate for error message
        return candidates[0]

    x0_path = None
    if "ic" in meta and isinstance(meta["ic"], dict) and "x0_path" in meta["ic"]:
        x0_path = base / meta["ic"]["x0_path"]                           # dataset schema
    elif "outputs" in meta and isinstance(meta["outputs"], dict) and meta["outputs"].get("x0"):
        x0_path = _resolve_x0_from_config(meta["outputs"], base)         # config schema
    else:
        # last-resort default in the same dir as the json
        x0_path = base / "x0.npy"

    if not x0_path.exists():
        raise SystemExit(f"Could not find x0 at: {x0_path}")

    # Originals X/t
    if "outputs" in meta and isinstance(meta["outputs"], dict) and \
       "X_path" in meta["outputs"] and "t_path" in meta["outputs"]:
        X_orig_path = base / meta["outputs"]["X_path"]                   # dataset schema
        t_orig_path = base / meta["outputs"]["t_path"]
    else:
        X_orig_path = base / "X.npy"                                     # config schema default
        t_orig_path = base / "t.npy"
    if not (X_orig_path.exists() and t_orig_path.exists()):
        raise SystemExit(f"Could not find originals at: {X_orig_path} and {t_orig_path}")

    # --- Load data ---
    x0 = np.load(x0_path).astype(np.float64, copy=False)
    X_orig = np.load(X_orig_path)           # keep original dtype (likely float32)
    t_orig = np.load(t_orig_path)           # keep original dtype (likely float32)
    
    # --- Re-simulate on the SAME grid (critical) ---
    t_re, X_re = simulate_l96(
        N, F, tmax, steps, x0,
        method=method, rtol=use_rtol, atol=use_atol,
        t_eval=None
    )

    # For deterministic RK4, compare at the saved precision (float32)
    deterministic = (method.upper() == "RK4FIXED")
    if deterministic:
        X_re_cmp = X_re.astype(np.float32, copy=False)
        t_re_cmp = t_re.astype(t_orig.dtype, copy=False)
    else:
        X_re_cmp = X_re
        t_re_cmp = t_re
    
    print("\n== Replay compare ==")
    print(f"Source: {jpath}")
    print(f"Params: N={N}, F={F}, tmax={tmax}, steps={steps}, method={method}, rtol={use_rtol}, atol={use_atol}")
    print("x0 vs X_orig[:,0] max|Δ| =", float(np.max(np.abs(x0 - X_orig[:, 0].astype(np.float64)))))
    print("x0 vs X_re[:,0]   max|Δ| =", float(np.max(np.abs(x0 - X_re[:, 0]))))
    
    ok_t, _ = _compare_arrays("t", t_orig, t_re_cmp, strict=True, rtol=0.0, atol=0.0)
    ok_X, _ = _compare_arrays("X", X_orig, X_re_cmp, strict=deterministic, rtol=rtol, atol=atol)

    return 0 if (ok_t and ok_X) else 1



def main():
    p = argparse.ArgumentParser(description="Lorenz-96 simulator → space-time image.")
    # dynamics
    p.add_argument("--N", type=int, default=40)
    p.add_argument("--F", type=float, default=8.0)
    p.add_argument("--tmax", type=float, default=10.0)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--method", type=str, default="RK45",
                   help='Integrator: "RK45", "DOP853", or "RK4FIXED" (deterministic fixed-step RK4)')
    p.add_argument("--rtol", type=float, default=1e-6)
    p.add_argument("--atol", type=float, default=1e-9)
    # initial conditions
    p.add_argument("--ic-mode", choices=["steady", "random"], default="steady")
    p.add_argument("--eps", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--ic-file", type=Path, default=None)
    # outputs
    p.add_argument("--cmap", type=str, default="RdBu_r")
    p.add_argument("--vmin", type=float, default=None)
    p.add_argument("--vmax", type=float, default=None)
    p.add_argument("--save", type=Path, default=None)
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--save-npy", type=Path, default=None)
    p.add_argument("--save-x0", type=Path, default=None)
    p.add_argument("--save-config", type=Path, default=None)
    p.add_argument("--no-show", action="store_true")
    # printing
    p.add_argument("--print-inputs", action="store_true")
    p.add_argument("--print-ic", choices=["summary", "full"], default="summary")
    # config replay
    p.add_argument("--from-config", type=Path, default=None,
                   help="Load all dynamics params + x0 from inputs_*.json and run")
    p.add_argument("--replay", type=Path, default=None,
                   help="Path to inputs_*.json OR to a dataset directory (use with --replay-id).")
    p.add_argument("--replay-id", type=int, default=None,
                   help="Sample id when --replay points to a directory.")
    p.add_argument("--replay-save", type=Path, default=None,
                   help="Directory to save replayed arrays (t.npy, X.npy).")
    p.add_argument("--replay-save-x0", type=Path, default=None,
                   help="Optional path to save the x0 used in replay (x0.npy).")
    p.add_argument("--replay-force-method", type=str, default=None,
                   help='Override method from JSON (e.g., "RK4FIXED" for deterministic equality).')
    p.add_argument("--replay-rtol", type=float, default=1e-6, help="allclose rtol for X (non-deterministic replays).")
    p.add_argument("--replay-atol", type=float, default=1e-6, help="allclose atol for X (non-deterministic replays).")
    args = p.parse_args()

    # replay mode - exit after replay from json input
    if args.replay is not None:
        code = replay(
            replay_path=args.replay,
            replay_id=args.replay_id,
            save_dir=args.replay_save,
            save_x0=args.replay_save_x0,
            force_method=args.replay_force_method,
            strict=False,  # exact equality enforced automatically if method == RK4FIXED
            rtol=args.replay_rtol,
            atol=args.replay_atol,
        )
        sys.exit(code)

    # Apply config overrides BEFORE simulation
    if args.from_config is not None:
        with open(args.from_config, "r") as f:
            meta = json.load(f)
        args.N     = int(meta["N"])
        args.F     = float(meta["F"])
        args.tmax  = float(meta["tmax"])
        args.steps = int(meta["steps"])
        args.method= meta.get("method", args.method)
        args.rtol  = float(meta.get("rtol", args.rtol))
        args.atol  = float(meta.get("atol", args.atol))
        base = args.from_config.parent
        args.ic_file = base / meta["ic"]["x0_path"]

    # Run simulation
    x0 = make_initial_conditions(args.N, args.F, args.ic_mode, args.eps, args.seed, args.ic_file)
    t, X = simulate_l96(args.N, args.F, args.tmax, args.steps, x0, args.method, args.rtol, args.atol)

    if args.print_inputs:
        print("\n== Lorenz-96 Inputs ==")
        print(f"N={args.N}, F={args.F}")
        print(f"tmax={args.tmax}, steps={args.steps}, method={args.method}, rtol={args.rtol}, atol={args.atol}")
        ic_desc = summarize_vector(x0, max_items=12) if args.print_ic == "summary" else np.array2string(
            x0, precision=6, separator=", ", max_line_width=120
        )
        print("x0 =", ic_desc)
        if args.ic_file:
            print(f"(x0 loaded from {args.ic_file})")
        elif args.seed is not None:
            print(f"(x0 generated with ic_mode={args.ic_mode}, eps={args.eps}, seed={args.seed})")
        else:
            print(f"(x0 generated with ic_mode={args.ic_mode}, eps={args.eps}, seed=None)")

    title = f"Lorenz-96  N={args.N}, F={args.F}, t∈[0,{args.tmax}], steps={args.steps}"
    fig = plot_spacetime(t, X, title, args.cmap, args.vmin, args.vmax)

    # Save outputs
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved image to: {args.save.resolve()}")

    if args.save_npy is not None:
        args.save_npy.mkdir(parents=True, exist_ok=True)
        np.save(args.save_npy / "t.npy", t.astype(np.float32, copy=False))
        np.save(args.save_npy / "X.npy", X.astype(np.float32, copy=False))
        print(f"Saved arrays to: {args.save_npy.resolve()}/(t.npy, X.npy)")
    if args.save_x0 is not None:
        args.save_x0.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save_x0, x0.astype(np.float64, copy=False))
        print(f"Saved initial condition to: {args.save_x0.resolve()}")
    if args.save_config is not None:
        args.save_config.parent.mkdir(parents=True, exist_ok=True)
        config = dict(
            N=args.N, F=args.F, tmax=args.tmax, steps=args.steps,
            method=args.method, rtol=args.rtol, atol=args.atol,
            ic=dict(mode=args.ic_mode, eps=args.eps, seed=args.seed,
                    source=str(args.ic_file) if args.ic_file else "generated"),
            visualization=dict(cmap=args.cmap, vmin=args.vmin, vmax=args.vmax, dpi=args.dpi),
            outputs=dict(image=str(args.save) if args.save else None,
                         arrays=str(args.save_npy) if args.save_npy else None,
                         x0=str(args.save_x0) if args.save_x0 else None)
        )
        with open(args.save_config, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Saved config to: {args.save_config.resolve()}")

    if not args.no_show:
        import matplotlib.pyplot as plt
        plt.show()

if __name__ == "__main__":
    main()

