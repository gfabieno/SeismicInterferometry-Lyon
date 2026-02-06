import numpy as np
import matplotlib.pyplot as plt

from straight_ray_tomography import (
    Grid2D, Path,
    all_cell_centers,
    build_G, ray_density,
    build_F, build_H, build_Q,
    solve_tomography, predict
)

# -----------------------------
# 1) Set up grid + true dv/v model
# -----------------------------
rng = np.random.default_rng(7)

Lx = 10_000.0
Ly = 10_000.0
dx = dy = 200.0
grid = Grid2D(x0=0.0, y0=0.0, dx=dx, dy=dy, nx=int(Lx/dx), ny=int(Ly/dy))

xc, yc = all_cell_centers(grid)
m_true = np.zeros(grid.ncell, float)

def gauss(x0, y0, amp, sig):
    d2 = (xc - x0)**2 + (yc - y0)**2
    return amp * np.exp(-0.5 * d2 / (sig*sig))

m_true += gauss(3500, 6500, +0.004, 900)    # +0.4%
m_true += gauss(7000, 3000, -0.003, 1200)   # -0.3%

# -----------------------------
# 2) Make synthetic stations + ray pairs
# -----------------------------
N_boundary = 16
angles = np.linspace(0, 2*np.pi, N_boundary, endpoint=False)
boundary = np.column_stack([0.5*(1+np.cos(angles))*Lx, 0.5*(1+np.sin(angles))*Ly])

N_inside = 14
inside = np.column_stack([rng.uniform(0.1*Lx, 0.9*Lx, N_inside),
                          rng.uniform(0.1*Ly, 0.9*Ly, N_inside)])

sta_xy = np.vstack([boundary, inside])
N = sta_xy.shape[0]

all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
rng.shuffle(all_pairs)
P = 250
pairs_ij = np.array(all_pairs[:P], dtype=int)

# Build geometry-only Path objects (t_obs/sigma_t are placeholders here)
paths = []
for (i, j) in pairs_ij:
    x1, y1 = sta_xy[i]
    x2, y2 = sta_xy[j]
    paths.append(Path(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                      t_obs=0.0, sigma_t=1.0))

# -----------------------------
# 3) Forward operator + synthetic data
# -----------------------------
U0 = 1.0  # for dv/v test, just use 1 so G_ij = l_ij
G = build_G(paths, grid, U0=np.array([U0]))

d_clean = (G @ m_true).ravel()

sigma_d = 2e-4
d_obs = d_clean + rng.normal(0.0, sigma_d, size=d_clean.shape)
Cd_inv = np.full(P, 1.0/(sigma_d*sigma_d), float)

# -----------------------------
# 4) Plot ray paths (one + all) and density
# -----------------------------
i0, j0 = pairs_ij[0]
x1, y1 = sta_xy[i0]
x2, y2 = sta_xy[j0]

plt.figure()
plt.plot(sta_xy[:, 0], sta_xy[:, 1], "o")
for (a, b) in pairs_ij:
    xa, ya = sta_xy[a]
    xb, yb = sta_xy[b]
    plt.plot([xa, xb], [ya, yb], linewidth=0.5)
plt.title("All ray paths and stations")
plt.xlabel("x"); plt.ylabel("y")
plt.axis("equal")
plt.show()

plt.figure()
plt.plot(sta_xy[:, 0], sta_xy[:, 1], "o")
plt.plot([x1, x2], [y1, y2], linewidth=2.0)
plt.title("Example ray path between two stations")
plt.xlabel("x"); plt.ylabel("y")
plt.axis("equal")
plt.show()

rho = ray_density(paths, grid, mode="count")
plt.figure()
plt.imshow(rho.reshape(grid.ny, grid.nx), origin="lower",
           extent=[grid.x0, grid.x1, grid.y0, grid.y1])
plt.title("Ray path density (count per cell)")
plt.xlabel("x"); plt.ylabel("y")
plt.colorbar()
plt.show()

# -----------------------------
# 5) Validate G matrix
# -----------------------------
irow = 0   # choose any ray index
g_row = G.getrow(irow)          # sparse row
cell_idx = g_row.indices        # cells crossed
cell_len = g_row.data           # l_ij / U0 (or l_ij if U0=1)

Gmap = np.zeros(grid.ncell)
Gmap[cell_idx] = cell_len        # weights along the ray

i, j = pairs_ij[irow]
x1, y1 = sta_xy[i]
x2, y2 = sta_xy[j]

plt.figure()
plt.imshow(
    Gmap.reshape(grid.ny, grid.nx),
    origin="lower",
    extent=[grid.x0, grid.x1, grid.y0, grid.y1]
)
plt.plot([x1, x2], [y1, y2], "r--", lw=1)  # overlay ray
plt.colorbar(label="G_ij (path length weight)")
plt.title(f"G row {irow} (should trace the ray)")
plt.xlabel("x"); plt.ylabel("y")
plt.show()

assert np.allclose(
    np.array(G.sum(axis=1)).ravel(),
    np.array([
        np.hypot(p.x2 - p.x1, p.y2 - p.y1) / U0
        for p in paths
    ]),
    rtol=1e-10
)
print("G matrix validation passed.")

# -----------------------------
# 5) Regularization + inversion
# -----------------------------
F = build_F(grid, sigma=100.0)             # smoothing length
H = build_H(rho, lam=0.02)                 # density-weighted damping
Q = build_Q(F, H, alpha=50.0, beta=0.5)     # weights

# Use direct solve for portability across SciPy versions
m_est, info = solve_tomography(G, d_obs, Cd_inv, Q, method="spsolve")

d_pred = predict(G, m_est)
resid = d_obs - d_pred

# -----------------------------
# 6) Compare true vs inverted
# -----------------------------
m_true_map = m_true.reshape(grid.ny, grid.nx)
m_est_map  = m_est.reshape(grid.ny, grid.nx)
diff_map   = (m_est - m_true).reshape(grid.ny, grid.nx)

plt.figure()
plt.imshow(m_true_map, origin="lower", extent=[grid.x0, grid.x1, grid.y0, grid.y1])
plt.title("True dv/v model (m_true)")
plt.xlabel("x"); plt.ylabel("y")
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(m_est_map, origin="lower", extent=[grid.x0, grid.x1, grid.y0, grid.y1])
plt.title("Inverted dv/v model (m_est)")
plt.xlabel("x"); plt.ylabel("y")
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(diff_map, origin="lower", extent=[grid.x0, grid.x1, grid.y0, grid.y1])
plt.title("Difference (m_est - m_true)")
plt.xlabel("x"); plt.ylabel("y")
plt.colorbar()
plt.show()

rmse = float(np.sqrt(np.mean((m_est - m_true)**2)))
corr = float(np.corrcoef(m_est, m_true)[0, 1])
print("Solver info:", info)
print("RMSE:", rmse)
print("Corr:", corr)
print("Residual RMS:", float(np.sqrt(np.mean(resid**2))))


# -----------------------------
# 6) Display resolution
# -----------------------------

from straight_ray_tomography import resolution_matrix_rows, resolution_diag_hutchinson

j0 = (grid.ny//2) * grid.nx + (grid.nx//2)
Rrow = resolution_matrix_rows(G, Cd_inv, Q, [j0], solver="spsolve")[j0]

plt.figure()
plt.imshow(Rrow.reshape(grid.ny, grid.nx),
           origin="lower",
           extent=[grid.x0, grid.x1, grid.y0, grid.y1])
plt.colorbar(label="Resolution amplitude")
plt.title(f"Resolution kernel (row {j0})")
plt.xlabel("x"); plt.ylabel("y")
plt.show()

print("Diagonal resolution at center cell:", Rrow[j0])


Rdiag = resolution_diag_hutchinson(G, Cd_inv, Q, 100, solver="spsolve")

plt.figure()
plt.imshow(Rdiag.reshape(grid.ny, grid.nx),
           origin="lower",
           extent=[grid.x0, grid.x1, grid.y0, grid.y1],
           vmin=0, vmax=1)
plt.colorbar(label="Resolution (diagonal)")
plt.title("Model resolution (R diagonal)")
plt.xlabel("x"); plt.ylabel("y")
plt.show()

print("Max diagonal resolution:", np.max(Rdiag))
print("Hutchinson diagonal resolution at center cell:", Rdiag[j0])