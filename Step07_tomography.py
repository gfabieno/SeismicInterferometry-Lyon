#!/usr/bin/env python3
"""
Apply straight-ray dv/v tomography to stretching dv/v measurements from Step05.

Inputs (from your pipeline):
  - dv_stretching.h5  (output of Step05_dvv_all_pairs.py)
    Contains:
      /starttimes        (D,)
      /pairs_i_j         (P,2)  station indices
      /stations          (N,)   station names
      /dvv_<side>        (D,P)  dv/v time series for each pair
      /cc_<side>         (D,P)  best correlation coefficient per dv/v estimate
      /coords           (N,2)  station coordinates (x,y)
    :contentReference[oaicite:3]{index=3}

Optional:
  - reference HDF5 used by Step05 (e.g., references_mean.h5)
    If it contains /window/t0_s and /window/t1_s per pair,
    we compute t_eff per pair as midpoint of that window.
    :contentReference[oaicite:4]{index=4}

Outputs:
  - One NPZ per day with fields: dvv_map (ny,nx), m (ncell,), residuals, etc.

Usage example:
  python apply_dvv_tomography.py \
      --dvv-h5 outputs/STEP05_dvv_all_pairs/dv_stretching.h5 \
      --meta-h5 outputs/beam_xcorr/meta.h5 \
      --ref-h5 outputs/beam_xcorr/references_mean.h5 \
      --window-side causal \
      --grid-dx 200 --grid-dy 200 \
      --U0 3000 \
      --sigma-smooth 500 \
      --alpha 1.0 --beta 1.0 --lam-density 0.05 \
      --out-dir outputs/STEP06_dvv_tomo
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import numpy as np
import h5py

from straight_ray_tomography import (
    Grid2D, Path as RayPath, invert_one_period
    , resolution_matrix_rows, resolution_diag_hutchinson
)
from matplotlib import pyplot as plt


# -------------------------
# IO helpers
# -------------------------

def _as_str_array(x) -> np.ndarray:
    x = np.asarray(x)
    if x.dtype.kind in ("S", "O"):
        return x.astype("U")
    return x


def read_step05_dvv(dvv_h5: Path, window_side: str) -> Dict[str, Any]:
    """Read dvv/cc arrays and geometry from Step05 output HDF5."""
    side = window_side.lower()
    with h5py.File(dvv_h5, "r") as h5:
        starttimes = _as_str_array(h5["starttimes"][...])
        pairs_ij = h5["pairs_i_j"][...].astype(np.int64)  # (P,2)
        stations = _as_str_array(h5["stations"][...])

        dvv_key = f"dvv_{side}"
        cc_key = f"cc_{side}"
        if dvv_key not in h5:
            raise KeyError(f"Missing dataset {dvv_key} in {dvv_h5}")
        dvv = h5[dvv_key][...].astype(np.float64)  # (D,P)
        cc = h5[cc_key][...].astype(np.float64)    # (D,P)
        # read station coordinates if present (expected as (N,2))
        if "coords" in h5:
            coords = h5["coords"][...].astype(np.float64)
        elif "meta/coords_xy_m" in h5:
            coords = h5["meta/coords_xy_m"][...].astype(np.float64)
        else:
            raise KeyError("Missing station coordinates ('coords' or 'meta/coords_xy_m') in Step05 HDF5")

    return dict(
        starttimes=starttimes,
        pairs_ij=pairs_ij,
        stations=stations,
        dvv=dvv,
        cc=cc,
        coords=coords,
        window_side=side,
    )


def read_ref_windows(ref_h5: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Read per-pair stretching windows (t0_s, t1_s) if present."""
    if ref_h5 is None:
        return None
    with h5py.File(ref_h5, "r") as h5:
        if "window" not in h5:
            return None
        g = h5["window"]
        if "t0_s" not in g or "t1_s" not in g:
            return None
        t0 = g["t0_s"][...].astype(np.float64)
        t1 = g["t1_s"][...].astype(np.float64)
    return t0, t1

# -------------------------
# Tomography-specific mapping
# -------------------------

def build_paths_for_geometry(
    pairs_ij: np.ndarray,
    coords_xy: np.ndarray,
    t_obs: np.ndarray,
    sigma_t: np.ndarray,
) -> list[RayPath]:
    """
    Make a list of RayPath objects. Geometry comes from station coords,
    while t_obs and sigma_t are provided per pair for a given day.
    """
    paths: list[RayPath] = []
    for pidx, (i, j) in enumerate(pairs_ij):
        x1, y1 = coords_xy[i]
        x2, y2 = coords_xy[j]
        paths.append(RayPath(
            x1=float(x1), y1=float(y1),
            x2=float(x2), y2=float(y2),
            t_obs=float(t_obs[pidx]),
            sigma_t=float(sigma_t[pidx]),
        ))
    return paths


def estimate_sigma_dt_from_cc(
    cc: np.ndarray,
    base_sigma: float,
    min_cc: float,
    max_sigma: float,
) -> np.ndarray:
    """
    Heuristic uncertainty model for dt derived from correlation coefficient.

    - clip cc >= min_cc (low cc => very uncertain)
    - sigma increases as cc decreases

    You can replace this with your preferred error model.
    """
    cc_clip = np.clip(cc, min_cc, 1.0)
    # simple monotonic mapping: sigma = base_sigma / cc
    sigma = base_sigma / cc_clip
    return np.clip(sigma, base_sigma, max_sigma)

def save_day_plots(plots_root: Path,
                   starttime: str,
                   grid: Grid2D,
                   coords_xy: np.ndarray,
                   pairs_use: np.ndarray,
                   inv: Dict[str, Any],
                   sigma_dt: np.ndarray,
                   solver: str,
                   rtol: float,
                   maxiter: int):
    """Create and save 4 diagnostic plots for one day:
       1) ray paths, 2) ray density, 3) resolution row at center,
       4) resolution diagonal (Hutchinson, 100 probes).
    """
    plots_dir = plots_root
    plots_dir.mkdir(parents=True, exist_ok=True)

    xy = coords_xy

    # 1) Ray paths
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(xy[:, 0], xy[:, 1], s=20, c="k")
        for (ia, ib) in pairs_use:
            xa, ya = xy[ia]
            xb, yb = xy[ib]
            ax.plot([xa, xb], [ya, yb], color="C0", linewidth=0.5, alpha=0.6)
        ax.set_aspect("equal")
        ax.set_title(f"Ray paths - {starttime}")
        ax.set_xlabel("X"); ax.set_ylabel("Y")
        fig.tight_layout()
        f1 = plots_dir / f"{starttime}_ray_paths.png"
        fig.savefig(f1, dpi=200)
        plt.close(fig)
    except Exception as e:
        print(f"Warning: ray paths plot failed for {starttime}: {e}")

    # 2) Ray density (from inv['rho'])
    try:
        rho_map = inv.get("rho")
        if rho_map is None:
            from straight_ray_tomography import ray_density
            paths = build_paths_for_geometry(pairs_use, coords_xy,
                                             t_obs=np.zeros(len(pairs_use)), sigma_t=np.ones(len(pairs_use)))
            rho_map = ray_density(paths, grid, mode="count")
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(rho_map.reshape(grid.ny, grid.nx), origin="lower",
                       extent=[grid.x0, grid.x1, grid.y0, grid.y1], cmap="viridis")
        ax.set_title(f"Ray density - {starttime}")
        ax.set_xlabel("X"); ax.set_ylabel("Y")
        fig.colorbar(im, ax=ax, label="ray count")
        fig.tight_layout()
        f2 = plots_dir / f"{starttime}_ray_density.png"
        fig.savefig(f2, dpi=200)
        plt.close(fig)
    except Exception as e:
        print(f"Warning: ray density plot failed for {starttime}: {e}")

    # 3) Resolution row at model center
    try:
        j0 = (grid.ny // 2) * grid.nx + (grid.nx // 2)
        Cd_inv_diag = 1.0 / (sigma_dt ** 2)
        Rrows = resolution_matrix_rows(inv["G"], Cd_inv_diag, inv["Q"],
                                      [j0], solver=solver, rtol=rtol, maxiter=maxiter)
        Rrow = Rrows[j0]
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(Rrow.reshape(grid.ny, grid.nx), origin="lower",
                       extent=[grid.x0, grid.x1, grid.y0, grid.y1], cmap="magma")
        ax.set_title(f"Resolution row (center) - {starttime}")
        ax.set_xlabel("X"); ax.set_ylabel("Y")
        fig.colorbar(im, ax=ax, label="resolution")
        fig.tight_layout()
        f3 = plots_dir / f"{starttime}_resolution_row.png"
        fig.savefig(f3, dpi=200)
        plt.close(fig)
    except Exception as e:
        print(f"Warning: resolution row plot failed for {starttime}: {e}")

    # 4) Resolution diagonal via Hutchinson (100 probes)
    try:
        Rdiag = resolution_diag_hutchinson(inv["G"], Cd_inv_diag, inv["Q"],
                                           nprobe=100, solver=solver, rtol=rtol, maxiter=maxiter, seed=0)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(Rdiag.reshape(grid.ny, grid.nx), origin="lower",
                       extent=[grid.x0, grid.x1, grid.y0, grid.y1], cmap="inferno", vmin=0, vmax=1)
        ax.set_title(f"Resolution diag (Hutchinson, 100) - {starttime}")
        ax.set_xlabel("X"); ax.set_ylabel("Y")
        fig.colorbar(im, ax=ax, label="diag(R)")
        fig.tight_layout()
        f4 = plots_dir / f"{starttime}_resolution_diag.png"
        fig.savefig(f4, dpi=200)
        plt.close(fig)
    except Exception as e:
        print(f"Warning: resolution diag plot failed for {starttime}: {e}")

    # dv/v model plot
    try:
        m = inv.get("m")
        if m is not None:
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(m.reshape(grid.ny, grid.nx), origin="lower",
                           extent=[grid.x0, grid.x1, grid.y0, grid.y1], cmap="seismic", vmin=-0.05, vmax=0.05)
            ax.set_title(f"dv/v model - {starttime}")
            ax.set_xlabel("X"); ax.set_ylabel("Y")
            fig.colorbar(im, ax=ax, label="dv/v")
            fig.tight_layout()
            f5 = plots_dir / f"{starttime}_dvv_model.png"
            fig.savefig(f5, dpi=200)
            plt.close(fig)
    except Exception as e:
        print(f"Warning: dv/v model plot failed for {starttime}: {e}")

# -------------------------
# Main
# -------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dvv-h5", default="outputs/STEP05_dvv_all_pairs/dv_stretching.h5",
                    type=Path)
    ap.add_argument("--ref-h5", type=Path, default=None)

    ap.add_argument("--window-side", type=str, default="sum",
                    choices=["causal", "acausal", "sum"])

    # Convert dvv->dt
    ap.add_argument("--default-velocity", type=float, default=400,
                    help="Velocity in m/s. Used if ref-h5 has no window/t0_s,t1_s.")
    ap.add_argument("--min-cc", type=float, default=0.8)
    ap.add_argument("--base-sigma-dt", type=float, default=0.02,
                    help="Seconds. Base dt uncertainty at cc~1.")
    ap.add_argument("--max-sigma-dt", type=float, default=1.0)

    # Grid definition
    ap.add_argument("--grid-dx", type=float, default=20 )
    ap.add_argument("--grid-dy", type=float, default=20)
    ap.add_argument("--pad", type=float, default=10,
                    help="Padding added around station bounding box, in same units as coords.")

    # Tomography parameters
    ap.add_argument("--U0", type=float, default=400.0,
                    help="Reference velocity used only for scaling G (const is fine).")
    ap.add_argument("--sigma-smooth", default=50.0,
                    type=float,
                    help="Gaussian smoothing length (same units as coords).")
    ap.add_argument("--rmax", type=float, default=None,
                    help="Kernel truncation radius. Default 3*sigma-smooth.")
    ap.add_argument("--alpha", type=float, default=50,
                    help="Weight for smoothing penalty F^T F.")
    ap.add_argument("--beta", type=float, default=0.005,
                    help="Weight for density damping H^T H.")
    ap.add_argument("--lam-density", type=float, default=0.02,
                    help="Lambda in exp(-lambda*rho) damping operator.")
    ap.add_argument("--density-mode", type=str, default="count", choices=["count", "length"])

    # Solve
    ap.add_argument("--solver", type=str, default="spsolve", choices=["cg", "spsolve"])
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--maxiter", type=int, default=2000)

    # Filtering / mask
    ap.add_argument("--min-pair-dist", type=float, default=0.0,
                    help="Drop pairs closer than this distance (same units as coords).")

    ap.add_argument("--min-pair-cc", type=float, default=None,
                    help="Drop pairs for a given day with cc < this value (per-day filtering). If omitted, no per-day pair removal is applied.")
    # Output
    ap.add_argument("--out-dir", default="outputs/STEP07_dvv_tomo",
                    type=Path)
    ap.add_argument("--save-every", type=int, default=1,
                    help="Save every Nth day (useful for long runs).")

    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load dvv time series from Step05 output ---
    step05 = read_step05_dvv(args.dvv_h5, args.window_side)
    starttimes = step05["starttimes"]    # (D,)
    pairs_ij = step05["pairs_ij"]        # (P,2)
    dvv = step05["dvv"]                  # (D,P)
    cc = step05["cc"]                    # (D,P)
    P = pairs_ij.shape[0]
    D = dvv.shape[0]
    stations = step05["stations"]  # (N,)
    coords_xy = step05["coords"]

    # --- Pair distances and optional filtering ---
    xy_i = coords_xy[pairs_ij[:, 0]]
    xy_j = coords_xy[pairs_ij[:, 1]]
    pair_dist = np.linalg.norm(xy_j - xy_i, axis=1)

    keep = np.ones(P, dtype=bool)
    if args.min_pair_dist > 0:
        keep &= (pair_dist >= float(args.min_pair_dist))

    if not np.all(keep):
        # filter all arrays by keep
        pairs_ij = pairs_ij[keep]
        pair_dist = pair_dist[keep]
        dvv = dvv[:, keep]
        cc = cc[:, keep]
        P = pairs_ij.shape[0]
        print(f"Filtered pairs: kept P={P}")

    # --- Build grid from station bounds ---
    xmin, ymin = coords_xy[:, 0].min(), coords_xy[:, 1].min()
    xmax, ymax = coords_xy[:, 0].max(), coords_xy[:, 1].max()
    xmin -= args.pad; ymin -= args.pad
    xmax += args.pad; ymax += args.pad

    nx = int(np.ceil((xmax - xmin) / args.grid_dx))
    ny = int(np.ceil((ymax - ymin) / args.grid_dy))
    grid = Grid2D(x0=float(xmin), y0=float(ymin),
                  dx=float(args.grid_dx), dy=float(args.grid_dy),
                  nx=int(nx), ny=int(ny))
    print(f"Grid: nx={grid.nx}, ny={grid.ny}, ncell={grid.ncell}")

    # --- Effective lapse time for dvv->dt mapping ---
    ref_windows = read_ref_windows(args.ref_h5) if args.ref_h5 is not None else None
    if ref_windows is not None:
        t0_s, t1_s = ref_windows
        t_eff = 0.5 * (t0_s + t1_s)
    else:
        t_eff = pair_dist / float(args.default_velocity)

    rho, Q, G = None, None, None  # precompute inside invert_one_period
    for di in range(D):
        if (di % args.save_every) != 0:
            continue

        dvv_i = dvv[di, :]  # (P,)
        cc_i = cc[di, :]    # (P,)
        # Per-day filtering by correlation if requested:
        if args.min_pair_cc is not None:
            # keep pairs that have finite dvv and cc >= threshold
            finite_mask = np.isfinite(dvv_i) & np.isfinite(cc_i)
            cc_mask = cc_i >= float(args.min_pair_cc)
            keep_day = finite_mask & cc_mask
            n_keep = int(np.sum(keep_day))
            if n_keep == 0:
                print(f"Day {starttimes[di]}: no pairs left after applying min-pair-cc={args.min_pair_cc} → skipping")
                continue
            if n_keep < dvv_i.size:
                print(f"Day {starttimes[di]}: keeping {n_keep}/{dvv_i.size} pairs after min-pair-cc={args.min_pair_cc}")
            # subset all per-pair arrays for this day
            pairs_use = pairs_ij[keep_day]
            pair_dist_use = pair_dist[keep_day]
            dvv_use = dvv_i[keep_day]
            cc_use = cc_i[keep_day]
            # effective time per pair
            t_eff_use = t_eff[keep_day]
        else:
            # use all pairs (after any global filtering applied earlier)
            pairs_use = pairs_ij
            pair_dist_use = pair_dist
            dvv_use = dvv_i
            cc_use = cc_i
            t_eff_use = t_eff

        # Convert dv/v -> dt for each selected pair:
        # dt = -(dvv)*t_eff
        # Then set t_obs = t_eff + dt = (1 - dvv)*t_eff
        t_obs = (1.0 - dvv_use) * t_eff_use
        # Data weights (only for the selected pairs)
        sigma_dt = estimate_sigma_dt_from_cc(
            cc=cc_use,
            base_sigma=float(args.base_sigma_dt),
            min_cc=float(args.min_cc),
            max_sigma=float(args.max_sigma_dt),
        )

        paths = build_paths_for_geometry(
            pairs_ij=pairs_use,
            coords_xy=coords_xy,
            t_obs=t_obs,
            sigma_t=sigma_dt,
        )

        inv = invert_one_period(paths, grid, args.default_velocity,
                               sigma_smooth=float(args.sigma_smooth),
                               rmax=float(args.rmax) if args.rmax is not None else None,
                               lam_density=float(args.lam_density),
                               alpha=float(args.alpha),
                               beta=float(args.beta),
                               density_mode=args.density_mode,
                               solver=args.solver,
                               maxiter=args.maxiter,
                               rtol=args.tol,
                               rho=rho, Q=Q, G=G)

        # ----- Produce requested diagnostic plots for this day -----
        # call helper to save plots (handles exceptions internally)
        save_day_plots(args.out_dir / "plots", starttimes[di], grid, coords_xy, pairs_use, inv, sigma_dt, args.solver, args.tol, args.maxiter)
        # ----- end plotting -----

        # Save
        out = args.out_dir / f"dvv_tomo_{starttimes[di]}.npz"
        np.savez_compressed(
            out,
            starttime=starttimes[di],                         # dv/v model (ncell,)
            **inv,
            grid_x=np.linspace(grid.x0, grid.x1, grid.nx),      # (nx,))
            grid_y=np.linspace(grid.y0, grid.y1, grid.ny),      # (ny,)
            pair_dist=pair_dist_use,                          # (P_used,)
            stations=stations,                                # (N,)
            pairs_ij=pairs_use,                               # (P_used,2)
            window_side=args.window_side,
        )
        print(f"Saved tomography result for day {starttimes[di]} to {out}")

    print("Done.")
    return 0





if __name__ == "__main__":
    raise SystemExit(main())
