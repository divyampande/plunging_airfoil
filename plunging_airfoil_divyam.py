import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon
from scipy.special import hankel2
from numba import njit, prange

# MASTER CONFIGURATION

# Core Physics
METHOD = "UDVM"  # 'LVM' or 'UDVM'
K_RED = 1  # Reduced frequency (k)
KH_VAL = 0.3  # Non-dimensional plunge velocity (kh = k * h)

# Features
USE_LAMB_OSEEN = True  # Desingularize with viscous core growth
USE_EULERIAN = True  # Project Lagrangian particles to Eulerian grid

# Run Modes
RUN_STANDARD = True
RUN_CINEMATIC_KH = False
RUN_COMPARE_METHODS = False
RUN_VALIDATE_THEODORSEN = False

# Simulation Specs
U_INF = 1.0
CHORD = 1.0
N_CYCLES = 5
STEPS_PER_CYCLE = 100
FPS = 30
N_PANELS = 32
WAKE_FRAC = 0.3
SIGMA_CORE = 0.05
NU_ARTIF = 2e-4

OUT_DIR = "results_master"
os.makedirs(OUT_DIR, exist_ok=True)

# GEOMETRY UTILITIES


def get_naca0012_coords(chord=1.0, n_points=50):
    beta = np.linspace(0, np.pi, n_points)
    x = 0.5 * chord * (1 - np.cos(beta))
    xn = x / chord
    yt = (
        5.0
        * chord
        * 0.12
        * (
            0.2969 * np.sqrt(xn)
            - 0.1260 * xn
            - 0.3516 * xn**2
            + 0.2843 * xn**3
            - 0.1015 * xn**4
        )
    )
    return np.concatenate([x, x[::-1]]), np.concatenate([yt, -yt[::-1]])


# NUMBA KERNELS


@njit(parallel=True, fastmath=True)
def biot_savart_kernel(
    target_x, target_y, source_x, source_y, gamma, sigma, use_lamb_oseen
):
    M, N = target_x.shape[0], source_x.shape[0]
    u, v = np.zeros(M), np.zeros(M)
    inv_2pi = 1.0 / (2.0 * np.pi)
    eps = 1e-10

    for i in prange(M):
        tx, ty = target_x[i], target_y[i]
        ui, vi = 0.0, 0.0
        for j in range(N):
            dx = tx - source_x[j]
            dy = ty - source_y[j]

            if use_lamb_oseen:
                r2 = dx * dx + dy * dy + eps
                if r2 < eps * 2:
                    continue
                sig2 = sigma[j] ** 2
                factor = (gamma[j] * inv_2pi / r2) * (1.0 - np.exp(-r2 / sig2))
            else:
                r2 = dx * dx + dy * dy + sigma[j] ** 2
                factor = gamma[j] * inv_2pi / r2

            ui -= factor * dy
            vi += factor * dx
        u[i] = ui
        v[i] = vi
    return u, v


@njit(parallel=True, fastmath=True)
def render_eulerian_field(X_grid, Y_grid, vx, vy, vg, vs, xmax):
    rows, cols = X_grid.shape
    omega = np.zeros((rows, cols), dtype=np.float64)
    N = vx.shape[0]
    inv_pi = 1.0 / np.pi
    inv_sig2 = 1.0 / (vs**2 + 1e-10)
    coeff = vg * inv_pi * inv_sig2
    cull_dist = 4.0 * vs

    for i in prange(rows):
        for j in range(cols):
            xg, yg = X_grid[i, j], Y_grid[i, j]
            val = 0.0
            for k in range(N):
                if vx[k] > xmax + 1.0:
                    continue
                dx = xg - vx[k]
                if np.abs(dx) > cull_dist[k]:
                    continue
                dy = yg - vy[k]
                if np.abs(dy) > cull_dist[k]:
                    continue

                r2_scaled = (dx * dx + dy * dy) * inv_sig2[k]
                if r2_scaled < 16.0:
                    val += coeff[k] * np.exp(-r2_scaled)
            omega[i, j] = val
    return omega


# UNIFIED SOLVER CLASS (LVM & UDVM)


class UnifiedVortexSolver:
    def __init__(self, method, k_red, kh, n_panels=N_PANELS, lamb_oseen=USE_LAMB_OSEEN):
        self.method = method.upper()
        self.c, self.U, self.lamb_oseen = CHORD, U_INF, lamb_oseen

        self.k = k_red
        self.h0 = kh / k_red
        self.omega = 2.0 * self.k * self.U / self.c
        self.f = self.omega / (2.0 * np.pi)
        self.dt = 1.0 / (self.f * STEPS_PER_CYCLE)

        self.N = 1 if self.method == "LVM" else n_panels
        self.dx = self.c / self.N
        self.x_nodes = np.linspace(0, self.c, self.N + 1)

        self.x_bv = self.x_nodes[:-1] + 0.25 * self.dx
        self.x_cp = self.x_nodes[:-1] + 0.75 * self.dx
        self.y_base = np.zeros(self.N)
        self.normals = np.column_stack((np.zeros(self.N), np.ones(self.N)))

        self.A_bb = np.zeros((self.N, self.N))
        core_sq = SIGMA_CORE**2 if self.method == "LVM" else 1e-10
        for i in range(self.N):
            for j in range(self.N):
                dx = self.x_cp[i] - self.x_bv[j]
                dy = 0.0
                r2 = dx**2 + dy**2 + core_sq
                v = (1.0 / (2.0 * np.pi * r2)) * dx
                self.A_bb[i, j] = v * self.normals[i, 1]

        self.wx, self.wy, self.wg, self.ws = (
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
        )
        self.Gb = np.zeros(self.N)

        self.t_hist, self.h_hist, self.Cl_hist, self.Ct_hist = [], [], [], []
        self.snapshots = []

    def solve_step(self, t):
        h = -self.h0 * np.cos(self.omega * t)
        h_dot = self.h0 * self.omega * np.sin(self.omega * t)
        h_ddot = self.h0 * self.omega**2 * np.cos(self.omega * t)

        y_cp_t = self.y_base + h
        y_bv_t = self.y_base + h
        xW = self.c + WAKE_FRAC * self.U * self.dt
        yW = h

        uw, vw = 0.0, 0.0
        if len(self.wx) > 0:
            uw, vw = biot_savart_kernel(
                self.x_cp, y_cp_t, self.wx, self.wy, self.wg, self.ws, self.lamb_oseen
            )

        RHS = -(self.U + uw) * self.normals[:, 0] - (vw - h_dot) * self.normals[:, 1]

        un, vn = biot_savart_kernel(
            self.x_cp,
            y_cp_t,
            np.array([xW]),
            np.array([yW]),
            np.array([1.0]),
            np.array([SIGMA_CORE]),
            self.lamb_oseen,
        )

        A_sys = np.zeros((self.N + 1, self.N + 1))
        A_sys[: self.N, : self.N] = self.A_bb
        A_sys[: self.N, self.N] = un * self.normals[:, 0] + vn * self.normals[:, 1]
        A_sys[self.N, : self.N] = 1.0
        A_sys[self.N, self.N] = 1.0

        b_sys = np.zeros(self.N + 1)
        b_sys[: self.N] = RHS
        b_sys[self.N] = np.sum(self.Gb)

        sol = np.linalg.solve(A_sys, b_sys)
        Gb_new, Gw_new = sol[: self.N], sol[self.N]

        Gamma_tot = np.sum(Gb_new)
        Gamma_dot = (Gamma_tot - np.sum(self.Gb)) / self.dt

        Cl_circ = -(2.0 * Gamma_tot) / (self.U * self.c) - (2.0 * Gamma_dot) / (
            self.U**2
        )
        Cl_nc = -(np.pi * self.c / 2.0) * (h_ddot / self.U**2)
        Cl_total = Cl_circ + Cl_nc

        alpha_eff = -h_dot / self.U
        Ct_total = Cl_total * np.sin(alpha_eff)

        if len(self.wx) > 0:
            ub, vb = biot_savart_kernel(
                self.wx,
                self.wy,
                self.x_bv,
                y_bv_t,
                Gb_new,
                np.full(self.N, SIGMA_CORE),
                self.lamb_oseen,
            )
            uw_self, vw_self = biot_savart_kernel(
                self.wx, self.wy, self.wx, self.wy, self.wg, self.ws, self.lamb_oseen
            )
            self.wx += (self.U + ub + uw_self) * self.dt
            self.wy += (vb + vw_self) * self.dt
            if self.lamb_oseen:
                self.ws = np.sqrt(self.ws**2 + 4.0 * NU_ARTIF * self.dt)

        self.wx = np.append(self.wx, xW)
        self.wy = np.append(self.wy, yW)
        self.wg = np.append(self.wg, Gw_new)
        self.ws = np.append(self.ws, SIGMA_CORE)

        self.Gb = Gb_new

        self.t_hist.append(t)
        self.h_hist.append(h)
        self.Cl_hist.append(Cl_total)
        self.Ct_hist.append(Ct_total)

        return h

    def simulate(self):
        print(
            f"Running {self.method} | kh={self.h0*self.k:.2f} | Lamb-Oseen={self.lamb_oseen}"
        )
        n_steps = int(N_CYCLES * STEPS_PER_CYCLE)
        snap_int = max(1, STEPS_PER_CYCLE // FPS)

        for step in range(n_steps):
            t = step * self.dt
            h = self.solve_step(t)
            if step % snap_int == 0:
                self.snapshots.append(
                    {
                        "t": t,
                        "h": h,
                        "wx": self.wx.copy(),
                        "wy": self.wy.copy(),
                        "wg": self.wg.copy(),
                        "ws": self.ws.copy(),
                    }
                )
        steps_in_cycle = int(STEPS_PER_CYCLE)
        if len(self.Ct_hist) >= steps_in_cycle:
            mean_Ct = np.mean(self.Ct_hist[-steps_in_cycle:])
        else:
            mean_Ct = np.mean(self.Ct_hist)

        regime = "THRUST" if mean_Ct > 0 else "DRAG"
        print(f"  -> Mean Ct (last cycle): {mean_Ct:+.4f} [{regime}]")


# ANALYTICAL VALIDATION


def theodorsen_function(k):
    if k == 0:
        return 1.0 + 0.0j
    H1 = hankel2(1, k)
    H0 = hankel2(0, k)
    return H1 / (H1 + 1j * H0)


def analytical_cl_plunge(k, h0, t, U, c):
    omega = 2.0 * k * U / c
    Ck = theodorsen_function(k)
    F, G = Ck.real, Ck.imag

    h_dot = h0 * omega * np.sin(omega * t)
    h_ddot = h0 * omega**2 * np.cos(omega * t)

    Cl_circ = 2 * np.pi * (F * (-h_dot) / U + G * (-h_ddot) / (U * omega))
    Cl_nc = -np.pi * c / (2 * U**2) * h_ddot
    return Cl_circ + Cl_nc


# VISUALIZATION SUITE


def plot_history(solver, prefix):
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(solver.t_hist, solver.Cl_hist, "b-", lw=1.5)
    axes[0].set_ylabel("$C_l$")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f"{solver.method} Dynamics (kh={solver.h0*solver.k:.2f})")

    axes[1].plot(solver.t_hist, solver.Ct_hist, "r-", lw=1.5)
    axes[1].axhline(0, color="k", alpha=0.3)
    axes[1].set_ylabel("$C_t$")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{prefix}_history.png", dpi=150)
    plt.close()


def plot_final_wake(solver, prefix):
    fig, ax = plt.subplots(figsize=(10, 4))
    snap = solver.snapshots[-1]

    if USE_EULERIAN:
        ax.set_facecolor("#06060A")
        X_grid, Y_grid = np.meshgrid(np.linspace(-0.5, 10, 200), np.linspace(-2, 2, 80))
        omega = render_eulerian_field(
            X_grid, Y_grid, snap["wx"], snap["wy"], snap["wg"], snap["ws"], 10
        )
        ax.imshow(
            omega,
            extent=[-0.5, 10, -2, 2],
            origin="lower",
            cmap="seismic",
            vmin=-5,
            vmax=5,
        )
        af_color = "#00FFFF"
    else:
        sz = 50 * np.abs(snap["wg"]) / (np.max(np.abs(snap["wg"])) or 1.0) + 2
        colors = ["#d62728" if g > 0 else "#1f77b4" for g in snap["wg"]]
        ax.scatter(snap["wx"], snap["wy"], c=colors, s=sz, alpha=0.6, edgecolors="none")
        af_color = "#333333"
        ax.set_xlim(-0.5, 10)
        ax.set_ylim(-2, 2)

    x_base, y_base = get_naca0012_coords(CHORD, 40)
    ax.fill(x_base, y_base + snap["h"], color=af_color, alpha=0.8)

    ax.set_aspect("equal")
    ax.set_title(f"Final Wake Snapshot ({solver.method})")
    plt.savefig(f"{OUT_DIR}/{prefix}_wake.png", dpi=150)
    plt.close()


def generate_animation(solvers, titles, filename):
    import matplotlib.pyplot as plt

    nc = len(solvers)
    bg_color = "#06060A" if USE_EULERIAN else "#FFFFFF"
    grid_color = "#2A2A3A" if USE_EULERIAN else "#E0E0E0"
    text_color = "#00FFFF" if USE_EULERIAN else "#000000"
    af_color = "#00FFFF" if USE_EULERIAN else "#333333"

    fig, axes = plt.subplots(nc, 1, figsize=(12, 3.5 * nc), facecolor=bg_color)
    if nc == 1:
        axes = [axes]

    imgs = []
    polygons = []
    x_base, y_base = get_naca0012_coords(CHORD, 40)
    X_grid, Y_grid = np.meshgrid(
        np.linspace(-0.5, 15, 250), np.linspace(-2.5, 2.5, 100)
    )

    for ax, title in zip(axes, titles):
        ax.set_facecolor(bg_color)
        ax.set_xlim(-0.5, 15)
        ax.set_ylim(-2.5, 2.5)
        ax.set_aspect("equal")
        ax.set_title(title, color=text_color, fontweight="bold", loc="left")
        ax.tick_params(colors=text_color)
        ax.grid(True, color=grid_color, alpha=0.5)

        if USE_EULERIAN:
            img = ax.imshow(
                np.zeros_like(X_grid),
                extent=[-0.5, 15, -2.5, 2.5],
                origin="lower",
                cmap="seismic",
                vmin=-5,
                vmax=5,
            )
            imgs.append(img)
        else:
            img = ax.scatter([], [], alpha=0.6, edgecolors="none")
            imgs.append(img)

        poly = Polygon(np.column_stack([x_base, y_base]), facecolor=af_color, alpha=0.9)
        ax.add_patch(poly)
        polygons.append(poly)

    def animate(i):
        returns = []
        for j, solver in enumerate(solvers):
            snap = solver.snapshots[i]

            polygons[j].set_xy(np.column_stack([x_base, y_base + snap["h"]]))

            if USE_EULERIAN:
                if len(snap["wx"]) > 0:
                    field = render_eulerian_field(
                        X_grid,
                        Y_grid,
                        snap["wx"],
                        snap["wy"],
                        snap["wg"],
                        snap["ws"],
                        15.0,
                    )
                    imgs[j].set_data(field)
            else:
                if len(snap["wx"]) > 0:
                    wg = snap["wg"]
                    sizes = 40 * np.abs(wg) / (np.max(np.abs(wg)) + 1e-10) + 10

                    colors = ["#d62728" if g > 0 else "#1f77b4" for g in wg]

                    imgs[j].set_offsets(np.column_stack([snap["wx"], snap["wy"]]))
                    imgs[j].set_sizes(sizes)
                    imgs[j].set_color(colors)

            returns.extend([polygons[j], imgs[j]])
        return returns

    anim = animation.FuncAnimation(
        fig, animate, frames=len(solvers[0].snapshots), interval=1000 / FPS, blit=True
    )
    anim.save(f"{OUT_DIR}/{filename}", writer="ffmpeg", fps=FPS)
    plt.close()


# EXECUTION ROUTINES

if __name__ == "__main__":
    if RUN_STANDARD:
        s = UnifiedVortexSolver(METHOD, K_RED, KH_VAL)
        s.simulate()
        pref = f"{METHOD}_kh{KH_VAL:.2f}"
        plot_history(s, pref)
        plot_final_wake(s, pref)
        generate_animation([s], [f"{METHOD} | kh = {KH_VAL:.2f}"], f"{pref}_anim.mp4")

    if RUN_CINEMATIC_KH:
        kh_cases = [0.10, 0.30, 0.70]
        titles = [
            f"{METHOD} | Drag (kh=0.10)",
            f"{METHOD} | Neutral (kh=0.30)",
            f"{METHOD} | Thrust (kh=0.70)",
        ]
        solvers = []
        for kh in kh_cases:
            s = UnifiedVortexSolver(METHOD, K_RED, kh)
            s.simulate()
            solvers.append(s)
        generate_animation(solvers, titles, f"{METHOD}_cinematic.mp4")

    if RUN_COMPARE_METHODS:
        methods = ["LVM", "UDVM"]
        titles = [f"{m} | kh={KH_VAL:.2f}" for m in methods]
        solvers = []
        for m in methods:
            s = UnifiedVortexSolver(m, K_RED, KH_VAL)
            s.simulate()
            solvers.append(s)
        generate_animation(solvers, titles, f"Method_Comparison_kh{KH_VAL:.2f}.mp4")

    if RUN_VALIDATE_THEODORSEN:
        methods = ["LVM", "UDVM"]
        solvers = [UnifiedVortexSolver(m, K_RED, KH_VAL) for m in methods]
        for s in solvers:
            s.simulate()

        t_ana = np.array(solvers[0].t_hist)
        cl_ana = [
            analytical_cl_plunge(K_RED, KH_VAL / K_RED, t, U_INF, CHORD) for t in t_ana
        ]

        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))
        plt.plot(t_ana, cl_ana, "k--", lw=2, label="Theodorsen (Analytical)")
        colors = ["b", "r"]
        for s, c in zip(solvers, colors):
            plt.plot(s.t_hist, s.Cl_hist, color=c, alpha=0.7, label=s.method)

        plt.xlabel("Time (s)")
        plt.ylabel("$C_l$")
        plt.title(f"Validation against Theodorsen (kh = {KH_VAL:.2f})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{OUT_DIR}/Validation_Theodorsen.png", dpi=150)
        plt.close()
