# straight_ray_tomo.py
# Straight-ray group-velocity tomography (2-D) with:
#   - Gaussian/exponential smoothing regularizer: F = I - W  (row-normalized Gaussian weights)
#   - Ray-density–weighted damping: H = diag(exp(-lambda * rho))
# Uses SciPy sparse matrices throughout.

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional, Dict, Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# -----------------------------
# Data structures
# -----------------------------

@dataclass(frozen=True)
class Grid2D:
    """Axis-aligned regular 2-D grid of rectangular cells.

    Coordinates are in any Cartesian system (e.g., local ENU meters).
    The grid spans [x0, x0+nx*dx] x [y0, y0+ny*dy].
    """
    x0: float
    y0: float
    dx: float
    dy: float
    nx: int
    ny: int

    @property
    def ncell(self) -> int:
        return int(self.nx * self.ny)

    @property
    def x1(self) -> float:
        return self.x0 + self.nx * self.dx

    @property
    def y1(self) -> float:
        return self.y0 + self.ny * self.dy


@dataclass(frozen=True)
class Path:
    """A straight ray/path between two points with observed group traveltime and uncertainty."""
    x1: float
    y1: float
    x2: float
    y2: float
    t_obs: float
    sigma_t: float


# -----------------------------
# Index helpers
# -----------------------------

def cell_index(ix: int, iy: int, grid: Grid2D) -> int:
    """Flatten (ix,iy) to j in [0, ncell). ix fastest."""
    return int(iy * grid.nx + ix)


def unravel_index(j: int, grid: Grid2D) -> Tuple[int, int]:
    """Unflatten j -> (ix,iy)."""
    iy, ix = divmod(int(j), grid.nx)
    return ix, iy


def cell_center_xy(j: int, grid: Grid2D) -> Tuple[float, float]:
    ix, iy = unravel_index(j, grid)
    return (grid.x0 + (ix + 0.5) * grid.dx, grid.y0 + (iy + 0.5) * grid.dy)


def all_cell_centers(grid: Grid2D) -> Tuple[np.ndarray, np.ndarray]:
    """Return flattened arrays of cell center coordinates (xc, yc), length ncell."""
    ix = np.arange(grid.nx)
    iy = np.arange(grid.ny)
    Xc = grid.x0 + (ix + 0.5) * grid.dx
    Yc = grid.y0 + (iy + 0.5) * grid.dy
    XX, YY = np.meshgrid(Xc, Yc, indexing="xy")
    return XX.ravel(), YY.ravel()


# -----------------------------
# Ray-cell intersections (2D DDA traversal)
# -----------------------------

def _clip_segment_to_box(x1: float, y1: float, x2: float, y2: float,
                         xmin: float, xmax: float, ymin: float, ymax: float) -> Optional[Tuple[float, float, float, float]]:
    """Liang-Barsky clipping. Returns clipped segment endpoints or None if outside."""
    dx = x2 - x1
    dy = y2 - y1

    p = np.array([-dx, dx, -dy, dy], dtype=float)
    q = np.array([x1 - xmin, xmax - x1, y1 - ymin, ymax - y1], dtype=float)

    u1, u2 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if pi == 0:
            if qi < 0:
                return None
        else:
            u = qi / pi
            if pi < 0:
                u1 = max(u1, u)
            else:
                u2 = min(u2, u)
            if u1 > u2:
                return None

    cx1, cy1 = x1 + u1 * dx, y1 + u1 * dy
    cx2, cy2 = x1 + u2 * dx, y1 + u2 * dy
    return cx1, cy1, cx2, cy2


def ray_cell_intersections(path: Path, grid: Grid2D) -> Tuple[np.ndarray, np.ndarray]:
    """Compute intersected cell indices and path lengths in each cell.

    Returns:
      cols: (K,) int cell indices
      lens: (K,) float lengths within each cell

    Notes:
      - Clips segment to grid bounding box first (no contribution outside).
      - Uses a robust 2D voxel traversal (Amanatides & Woo style) in metric space.
    """
    clipped = _clip_segment_to_box(path.x1, path.y1, path.x2, path.y2,
                                  grid.x0, grid.x1, grid.y0, grid.y1)
    if clipped is None:
        return np.empty(0, dtype=int), np.empty(0, dtype=float)

    x1, y1, x2, y2 = clipped
    dx = x2 - x1
    dy = y2 - y1
    L = float(np.hypot(dx, dy))
    if L == 0.0:
        return np.empty(0, dtype=int), np.empty(0, dtype=float)

    # Determine starting cell
    # Clamp to valid indices (point can lie on max boundary after clipping)
    eps = 1e-12
    px = min(max(x1, grid.x0 + eps), grid.x1 - eps)
    py = min(max(y1, grid.y0 + eps), grid.y1 - eps)

    ix = int((px - grid.x0) // grid.dx)
    iy = int((py - grid.y0) // grid.dy)

    # Step direction
    step_x = 1 if dx > 0 else (-1 if dx < 0 else 0)
    step_y = 1 if dy > 0 else (-1 if dy < 0 else 0)

    # Parametric t along segment [0,1]
    # Next boundary in x and y
    def x_boundary(i: int) -> float:
        return grid.x0 + (i + (1 if step_x > 0 else 0)) * grid.dx

    def y_boundary(i: int) -> float:
        return grid.y0 + (i + (1 if step_y > 0 else 0)) * grid.dy

    # tMax: t at which we cross the first vertical/horizontal boundary
    if step_x != 0:
        tx_max = (x_boundary(ix) - x1) / dx
        tx_delta = grid.dx / abs(dx)
    else:
        tx_max = np.inf
        tx_delta = np.inf

    if step_y != 0:
        ty_max = (y_boundary(iy) - y1) / dy
        ty_delta = grid.dy / abs(dy)
    else:
        ty_max = np.inf
        ty_delta = np.inf

    # Traversal
    cols: List[int] = []
    lens: List[float] = []

    t = 0.0
    while 0 <= ix < grid.nx and 0 <= iy < grid.ny:
        j = cell_index(ix, iy, grid)

        # Next crossing param
        t_next = min(tx_max, ty_max, 1.0)
        if t_next < t:
            # Numerical safety
            t_next = t

        seg_len = (t_next - t) * L
        if seg_len > 0:
            cols.append(j)
            lens.append(seg_len)

        if t_next >= 1.0:
            break

        # Step across boundary
        if tx_max < ty_max:
            ix += step_x
            tx_max += tx_delta
        else:
            iy += step_y
            ty_max += ty_delta

        t = t_next

    return np.asarray(cols, dtype=int), np.asarray(lens, dtype=float)


# -----------------------------
# Build forward operator G and data vector d
# -----------------------------

def build_G(paths: Iterable[Path], grid: Grid2D, U0: np.ndarray) -> sp.csr_matrix:
    """Build sparse design matrix G with G_ij = l_ij / U0_j."""
    U0 = np.asarray(U0, dtype=float).ravel()
    if U0.size == 1:
        U0 = np.full(grid.ncell, float(U0[0]), dtype=float)
    if U0.size != grid.ncell:
        raise ValueError("U0 must be scalar or length ncell.")

    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []

    for i, p in enumerate(paths):
        c, l = ray_cell_intersections(p, grid)
        if c.size == 0:
            continue
        rows.extend([i] * c.size)
        cols.extend(c.tolist())
        data.extend((l / U0[c]).tolist())

    ndata = sum(1 for _ in paths)
    # NOTE: paths is an iterable; above loop consumed it if it wasn't a list.
    # So enforce list input at API level OR convert once here.
    # We'll handle robustly by requiring a list in the public runner; for low-level function:
    if not isinstance(paths, list):
        raise TypeError("build_G expects 'paths' to be a list (iterable would be consumed).")

    ndata = len(paths)
    G = sp.coo_matrix((np.array(data, float), (np.array(rows, int), np.array(cols, int))),
                      shape=(ndata, grid.ncell)).tocsr()
    return G


def build_data_vector(paths: List[Path], grid: Grid2D, U0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute d = t_obs - t0, and Cd_inv_diag = 1/sigma_t^2."""
    U0 = np.asarray(U0, dtype=float).ravel()
    if U0.size == 1:
        U0 = np.full(grid.ncell, float(U0[0]), dtype=float)
    if U0.size != grid.ncell:
        raise ValueError("U0 must be scalar or length ncell.")

    d = np.zeros(len(paths), dtype=float)
    Cd_inv = np.zeros(len(paths), dtype=float)

    for i, p in enumerate(paths):
        c, l = ray_cell_intersections(p, grid)
        t0 = 0.0 if c.size == 0 else float(np.sum(l / U0[c]))
        d[i] = float(p.t_obs - t0)
        if p.sigma_t <= 0:
            raise ValueError("sigma_t must be > 0 for all paths.")
        Cd_inv[i] = 1.0 / float(p.sigma_t ** 2)

    return d, Cd_inv


def ray_density(paths: List[Path], grid: Grid2D, mode: str = "count") -> np.ndarray:
    """Compute rho per cell.

    mode:
      - "count": number of rays that intersect the cell at least once
      - "length": total path length summed in the cell (optional alternative)
    """
    rho = np.zeros(grid.ncell, dtype=float)

    for p in paths:
        c, l = ray_cell_intersections(p, grid)
        if c.size == 0:
            continue
        if mode == "count":
            rho[np.unique(c)] += 1.0
        elif mode == "length":
            # sum lengths per visited cell
            # if duplicates occur (rare here), sum them
            for jj, ll in zip(c, l):
                rho[jj] += float(ll)
        else:
            raise ValueError("mode must be 'count' or 'length'.")

    return rho


# -----------------------------
# Regularization: Gaussian smoothing + ray-density damping
# -----------------------------

def build_gaussian_W(grid: Grid2D, sigma: float, rmax: Optional[float] = None) -> sp.csr_matrix:
    """Row-normalized Gaussian weight matrix W.

    W_jk ∝ exp(-d_jk^2 / (2 sigma^2)), truncated to neighbors within rmax.
    Rows are normalized so sum_k W_jk = 1.

    Choose:
      rmax ~ 3*sigma (default) to keep W sparse.
    """
    if sigma <= 0:
        raise ValueError("sigma must be > 0.")
    if rmax is None:
        rmax = 3.0 * sigma
    if rmax <= 0:
        raise ValueError("rmax must be > 0.")

    # neighbor range in indices
    rx = int(np.ceil(rmax / grid.dx))
    ry = int(np.ceil(rmax / grid.dy))

    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []

    # Precompute center coords in grid index space to avoid repeated calls
    xc, yc = all_cell_centers(grid)

    for j in range(grid.ncell):
        xj, yj = xc[j], yc[j]
        ix, iy = unravel_index(j, grid)

        # candidate neighbor box in index space
        ix0 = max(0, ix - rx)
        ix1 = min(grid.nx - 1, ix + rx)
        iy0 = max(0, iy - ry)
        iy1 = min(grid.ny - 1, iy + ry)

        js: List[int] = []
        ws: List[float] = []

        for nny in range(iy0, iy1 + 1):
            for nnx in range(ix0, ix1 + 1):
                k = cell_index(nnx, nny, grid)
                dx = xj - xc[k]
                dy = yj - yc[k]
                d2 = dx * dx + dy * dy
                if d2 <= rmax * rmax + 1e-15:
                    w = float(np.exp(-0.5 * d2 / (sigma * sigma)))
                    js.append(k)
                    ws.append(w)

        if not ws:
            # should not happen because j itself is always within rmax
            js = [j]
            ws = [1.0]

        ws = np.asarray(ws, float)
        ws /= ws.sum()

        rows.extend([j] * len(js))
        cols.extend(js)
        data.extend(ws.tolist())

    W = sp.coo_matrix((np.array(data, float), (np.array(rows, int), np.array(cols, int))),
                      shape=(grid.ncell, grid.ncell)).tocsr()
    return W


def build_F(grid: Grid2D, sigma: float, rmax: Optional[float] = None) -> sp.csr_matrix:
    """Smoothing operator F = I - W (W row-normalized Gaussian weights)."""
    W = build_gaussian_W(grid, sigma=sigma, rmax=rmax)
    I = sp.identity(grid.ncell, format="csr")
    return (I - W).tocsr()


def build_H(rho: np.ndarray, lam: float) -> sp.csr_matrix:
    """Ray-density damping operator H = diag(exp(-lam * rho))."""
    rho = np.asarray(rho, float).ravel()
    if lam < 0:
        raise ValueError("lam should be >= 0.")
    h = np.exp(-lam * rho)
    return sp.diags(h, offsets=0, format="csr")


def build_Q(F: sp.csr_matrix, H: sp.csr_matrix, alpha: float, beta: float) -> sp.csr_matrix:
    """Regularization matrix Q = alpha * F^T F + beta * H^T H."""
    if alpha < 0 or beta < 0:
        raise ValueError("alpha and beta must be >= 0.")
    Q = (alpha * (F.T @ F)) + (beta * (H.T @ H))
    return Q.tocsr()


# -----------------------------
# Solve
# -----------------------------

def solve_tomography(
    G: sp.csr_matrix,
    d: np.ndarray,
    Cd_inv_diag: np.ndarray,
    Q: sp.csr_matrix,
    method: str = "cg",
    rtol: float = 1e-6,
    maxiter: int = 2000,
    verbose: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Solve (G^T Cd^-1 G + Q) m = G^T Cd^-1 d.

    method: "cg" (recommended) or "spsolve" (direct).
    Returns:
      m, info_dict
    """
    d = np.asarray(d, float).ravel()
    w = np.asarray(Cd_inv_diag, float).ravel()
    if d.shape[0] != G.shape[0]:
        raise ValueError("d length must match number of rows in G.")
    if w.shape[0] != G.shape[0]:
        raise ValueError("Cd_inv_diag length must match number of rows in G.")

    # Apply Cd^-1 as row-scaling (diagonal): (Cd^-1 G) and (Cd^-1 d)
    Wd = sp.diags(w, offsets=0, format="csr")
    GTWd = G.T @ Wd
    A = (GTWd @ G + Q).tocsr()
    b = (GTWd @ d)

    info: Dict[str, Any] = {"method": method, "rtol": rtol, "maxiter": maxiter}

    if method.lower() == "spsolve":
        m = spla.spsolve(A, b)
        info["status"] = "direct"
        return np.asarray(m, float).ravel(), info

    if method.lower() == "cg":
        # Optional simple diagonal preconditioner
        M_inv = 1.0 / np.maximum(A.diagonal(), 1e-12)
        M = spla.LinearOperator(A.shape, matvec=lambda x: M_inv * x)

        it_count = {"k": 0}

        def _cb(_x):
            it_count["k"] += 1

        m, code = spla.cg(A, b, rtol=rtol, maxiter=maxiter, M=M, callback=_cb)
        info["iterations"] = it_count["k"]
        info["cg_code"] = code  # 0 success, >0 no convergence, <0 breakdown
        if verbose:
            print(f"[solve_tomography] cg_code={code}, iters={it_count['k']}")
        return np.asarray(m, float).ravel(), info

    raise ValueError("method must be 'cg' or 'spsolve'.")


def model_to_velocity(m: np.ndarray, U0: np.ndarray) -> np.ndarray:
    """Recover group velocity U from m=(U0-U)/U => U=U0/(1+m)."""
    m = np.asarray(m, float).ravel()
    U0 = np.asarray(U0, float).ravel()
    if U0.size == 1:
        U0 = np.full_like(m, float(U0[0]))
    if U0.size != m.size:
        raise ValueError("U0 must be scalar or same size as m.")
    return U0 / (1.0 + m)


def predict(G: sp.csr_matrix, m: np.ndarray) -> np.ndarray:
    """Predicted data: d_pred = G m."""
    return (G @ np.asarray(m, float).ravel()).ravel()


# -----------------------------
# High-level runner (one period)
# -----------------------------

def invert_one_period(
    paths: List[Path],
    grid: Grid2D,
    U0: np.ndarray,
    sigma_smooth: float,
    lam_density: float,
    alpha: float,
    beta: float,
    rmax: Optional[float] = None,
    density_mode: str = "count",
    solver: str = "spsolve",
    rtol: float = 1e-6,
    maxiter: int = 2000,
    verbose: bool = False,
    rho = None,
    Q = None,
    G = None,
) -> Dict[str, Any]:
    """End-to-end inversion for a single period.

    Returns dict with:
      m, U, d, d_pred, residual, rho, G, Q, solver_info
    """
    if not isinstance(paths, list):
        paths = list(paths)

    if G is None:
        G = build_G(paths, grid, U0)
    if rho is None:
        rho = ray_density(paths, grid, mode=density_mode)
    if Q is None:
        F = build_F(grid, sigma=sigma_smooth, rmax=rmax)
        H = build_H(rho, lam=lam_density)
        Q = build_Q(F, H, alpha=alpha, beta=beta)
    d, Cd_inv = build_data_vector(paths, grid, U0)

    m, sinfo = solve_tomography(G, d, Cd_inv, Q,
                                method=solver, rtol=rtol, maxiter=maxiter,
                                verbose=verbose)
    U = model_to_velocity(m, U0)

    d_pred = predict(G, m)
    r = d - d_pred

    return {
        "m": m,
        "U": U,
        "d": d,
        "d_pred": d_pred,
        "residual": r,
        "rho": rho,
        "G": G,
        "Q": Q,
        "solver_info": str(sinfo),
    }


def resolution_matrix_rows(G, Cd_inv_diag, Q, rows, solver="cg", rtol=1e-6, maxiter=2000):
    """
    Compute selected rows of the resolution matrix R.

    R = A^{-1} (G^T C_d^{-1} G)
    where A = G^T C_d^{-1} G + Q

    Parameters
    ----------
    rows : iterable of int
        Model indices (cells) for which to compute R[row, :]

    Returns
    -------
    R_rows : dict
        keys = row index, values = dense 1D numpy arrays (length nmodel)
    """
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    w = Cd_inv_diag
    Wd = sp.diags(w, 0, format="csr")
    GTWd = G.T @ Wd
    A = (GTWd @ G + Q).tocsr()

    R_rows = {}

    for j in rows:
        rhs = GTWd @ G[:, j]
        if solver == "cg":
            x, _ = spla.cg(A, rhs, rtol=rtol, maxiter=maxiter)
        else:
            x = spla.spsolve(A, rhs)
        R_rows[j] = x

    return R_rows


def resolution_diag_hutchinson(G, Cd_inv_diag, Q, nprobe=50, solver="spsolve", rtol=1e-6, maxiter=2000, seed=0):
    """
    Fast stochastic estimate of diag(R), where R = A^{-1} B,
      A = G^T C_d^{-1} G + Q
      B = G^T C_d^{-1} G

    Uses Hutchinson estimator:
      diag(R) ≈ (1/K) Σ_k  z_k ⊙ x_k
    where:
      u_k = B z_k
      solve A x_k = u_k
      z_k are Rademacher (+/-1) vectors.

    Returns
    -------
    rdiag : (nmodel,) ndarray
        Approximate diagonal of the resolution matrix.
    """
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    w = np.asarray(Cd_inv_diag, float).ravel()
    Wd = sp.diags(w, 0, format="csr")
    GTWd = G.T @ Wd
    B = (GTWd @ G).tocsr()
    A = (B + Q).tocsr()

    nmodel = A.shape[0]
    rng = np.random.default_rng(seed)
    acc = np.zeros(nmodel, dtype=float)

    if solver == "cg":
        # simple diagonal preconditioner
        M_inv = 1.0 / np.maximum(A.diagonal(), 1e-12)
        M = spla.LinearOperator(A.shape, matvec=lambda x: M_inv * x)

    for _ in range(int(nprobe)):
        z = rng.choice([-1.0, 1.0], size=nmodel)
        u = B @ z
        if solver == "cg":
            x, code = spla.cg(A, u, M=M, rtol=rtol, maxiter=maxiter)
            if code != 0:
                raise RuntimeError(f"cg did not converge (code={code})")
        else:
            x = spla.spsolve(A, u)
        acc += z * x

    return acc / float(nprobe)
