"""
=============================================================================
Method: Lumped Vortex Method (LVM)
Motion: Plunging Only
Geometry: Flat Plate (1 bound vortex, 1 collocation point)
=============================================================================
Description:
A standalone, plunge-only extraction of the dynamic boundary condition LVM solver.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon
from numba import njit, prange

# ============================================================================
# USER CONFIGURATION: CHOOSE YOUR WAKE REGIME HERE
# ============================================================================
# CASE 3: THRUST JET (kh = 0.80)
k_paper = 1.0
h_paper = 0.8

# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================
U_INF = 1.0
CHORD = 1.0
N_CYCLES = 5
FPS = 30
STEPS_PER_CYCLE = 100


# ============================================================================
# NACA 0012 AIRFOIL GEOMETRY
# ============================================================================
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


# ============================================================================
# NUMBA-OPTIMIZED AERODYNAMIC FUNCTIONS
# ============================================================================
@njit(fastmath=True)
def biot_savart(xp, yp, xv, yv, gamma, r_core=0.01):
    dx = xp - xv
    dy = yp - yv
    r2 = dx * dx + dy * dy + r_core * r_core
    fac = gamma / (2.0 * np.pi * r2)
    return -fac * dy, fac * dx


@njit(fastmath=True)
def wake_induced_velocity(xp, yp, wx, wy, wg, r_core=0.01):
    u_tot, v_tot = 0.0, 0.0
    for i in range(len(wx)):
        u, v = biot_savart(xp, yp, wx[i], wy[i], wg[i], r_core)
        u_tot += u
        v_tot += v
    return u_tot, v_tot


@njit(fastmath=True, parallel=True)
def fast_convect_wake(wx, wy, wg, xv, yv, Gamma_bound, U, dt, r_core):
    n = len(wx)
    new_x = np.zeros(n)
    new_y = np.zeros(n)
    for i in prange(n):
        ui, vi = U, 0.0
        ub, vb = biot_savart(wx[i], wy[i], xv, yv, Gamma_bound, r_core)
        ui += ub
        vi += vb
        for j in range(n):
            if j != i:
                uw, vw = biot_savart(wx[i], wy[i], wx[j], wy[j], wg[j], r_core)
                ui += uw
                vi += vw
        new_x[i] = wx[i] + ui * dt
        new_y[i] = wy[i] + vi * dt
    return new_x, new_y


# ============================================================================
# LUMPED VORTEX METHOD SOLVER (PLUNGE ONLY)
# ============================================================================
class PlungeVortexSolver:
    def __init__(self, k_paper, h_paper, include_added_mass=True):
        self.c = CHORD
        self.U = U_INF
        self.x0 = 0.75 * self.c
        self.r_core = 0.05 * self.c
        self.include_added_mass = include_added_mass

        self.k = k_paper
        self.omega = self.k * self.U / self.c
        self.f = self.omega / (2.0 * np.pi)
        self.dt = 1.0 / (self.f * STEPS_PER_CYCLE)

        self.h0 = h_paper * self.c

        self.wake_x = []
        self.wake_y = []
        self.wake_gamma = []

        self.Gamma_bound_history = []
        self.wake_snapshots = []
        self.tau_hist = []
        self.a_eff_hist = []
        self.Cl_hist = []
        self.Ct_hist = []

    def kinematics(self, t):
        h = self.h0 * np.sin(self.omega * t)
        h_dot = self.h0 * self.omega * np.cos(self.omega * t)
        h_ddot = -self.h0 * self.omega**2 * np.sin(self.omega * t)
        return h, h_dot, h_ddot, 0.0, 0.0, 0.0

    def get_geometry(self, h, alpha):
        pivot = self.x0
        y_trans = -h

        xv_local, yv_local = 0.25 * self.c - pivot, 0.0
        xv = pivot + xv_local * np.cos(alpha) + yv_local * np.sin(alpha)
        yv = y_trans - xv_local * np.sin(alpha) + yv_local * np.cos(alpha)

        xc_local, yc_local = 0.75 * self.c - pivot, 0.0
        xc = pivot + xc_local * np.cos(alpha) + yc_local * np.sin(alpha)
        yc = y_trans - xc_local * np.sin(alpha) + yc_local * np.cos(alpha)

        xte_local, yte_local = self.c - pivot, 0.0
        x_te = pivot + xte_local * np.cos(alpha) + yte_local * np.sin(alpha)
        y_te = y_trans - xte_local * np.sin(alpha) + yte_local * np.cos(alpha)

        nx, ny = np.sin(alpha), np.cos(alpha)
        return xv, yv, xc, yc, x_te, y_te, nx, ny

    def solve_step(self, t):
        h, h_dot, h_ddot, alpha, alpha_dot, alpha_ddot = self.kinematics(t)
        xv, yv, xc, yc, x_te, y_te, nx, ny = self.get_geometry(h, alpha)

        x_shed = x_te + 0.3 * self.U * self.dt * np.cos(alpha)
        y_shed = y_te - 0.3 * self.U * self.dt * np.sin(alpha)

        w_kin = (
            self.U * np.sin(alpha)
            + h_dot * np.cos(alpha)
            + alpha_dot * (0.75 * self.c - self.x0)
        )

        ub, vb = biot_savart(xc, yc, xv, yv, 1.0, self.r_core)
        K_bound = ub * nx + vb * ny

        us, vs = biot_savart(xc, yc, x_shed, y_shed, 1.0, self.r_core)
        K_shed = us * nx + vs * ny

        V_n_wake = 0.0
        if len(self.wake_x) > 0:
            uw, vw = wake_induced_velocity(
                xc,
                yc,
                np.array(self.wake_x, dtype=np.float64),
                np.array(self.wake_y, dtype=np.float64),
                np.array(self.wake_gamma, dtype=np.float64),
                self.r_core,
            )
            V_n_wake = uw * nx + vw * ny

        Gamma_prev = self.Gamma_bound_history[-1] if self.Gamma_bound_history else 0.0
        Gamma_bound = (-w_kin - V_n_wake - K_shed * Gamma_prev) / (K_bound - K_shed)
        Gamma_shed = Gamma_prev - Gamma_bound

        alpha_eff = (
            alpha + h_dot / self.U + (alpha_dot / self.U) * (0.75 * self.c - self.x0)
        )

        return (
            Gamma_bound,
            Gamma_shed,
            x_shed,
            y_shed,
            xv,
            yv,
            h,
            alpha,
            alpha_eff,
            h_dot,
            h_ddot,
            alpha_dot,
            alpha_ddot,
        )

    def compute_forces(
        self, Gamma_bound, alpha_eff, h_dot, h_ddot, alpha, alpha_dot, alpha_ddot
    ):
        Gamma_prev = self.Gamma_bound_history[-1] if self.Gamma_bound_history else 0.0
        Gamma_dot = (Gamma_bound - Gamma_prev) / self.dt

        Cl_circ = -(2.0 * Gamma_bound) / (self.U * self.c) - (2.0 * Gamma_dot) / (
            self.U**2
        )
        Cl_nc = 0.0
        if self.include_added_mass:
            Cl_nc = -(np.pi * self.c / 2.0) * (
                h_ddot / self.U**2
                + alpha_dot / self.U
                + (self.c / 2.0 - self.x0) * alpha_ddot / self.U**2
            )
        Cl_total = Cl_circ + Cl_nc
        Ct = Cl_total * np.sin(alpha_eff)
        return Cl_total, Ct

    def convect_wake(self, xv, yv, Gamma_bound):
        if not self.wake_x:
            return
        new_x, new_y = fast_convect_wake(
            np.array(self.wake_x, dtype=np.float64),
            np.array(self.wake_y, dtype=np.float64),
            np.array(self.wake_gamma, dtype=np.float64),
            float(xv),
            float(yv),
            float(Gamma_bound),
            float(self.U),
            float(self.dt),
            float(self.r_core),
        )
        self.wake_x, self.wake_y = new_x.tolist(), new_y.tolist()

    def simulate(self, n_cycles):
        n_steps = int(n_cycles * (1.0 / self.f) / self.dt)
        snapshot_interval = max(1, int(STEPS_PER_CYCLE / FPS))

        for step in range(n_steps):
            t = step * self.dt
            tau = 2.0 * self.U * t / self.c
            (
                Gamma_b,
                Gamma_s,
                x_s,
                y_s,
                xv,
                yv,
                h,
                alpha,
                a_eff,
                h_dot,
                h_ddot,
                alpha_dot,
                alpha_ddot,
            ) = self.solve_step(t)
            Cl, Ct = self.compute_forces(
                Gamma_b, a_eff, h_dot, h_ddot, alpha, alpha_dot, alpha_ddot
            )

            self.tau_hist.append(tau)
            self.a_eff_hist.append(a_eff)
            self.Cl_hist.append(Cl)
            self.Ct_hist.append(Ct)
            self.Gamma_bound_history.append(Gamma_b)

            if abs(Gamma_s) > 1e-10:
                self.wake_x.append(x_s)
                self.wake_y.append(y_s)
                self.wake_gamma.append(Gamma_s)

            self.convect_wake(xv, yv, Gamma_b)

            if step % snapshot_interval == 0:
                self.wake_snapshots.append(
                    {
                        "t": t,
                        "h": -h,
                        "alpha": alpha,
                        "wx": list(self.wake_x),
                        "wy": list(self.wake_y),
                        "wg": list(self.wake_gamma),
                    }
                )


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================
def plot_history(sim, kh_val, outdir):
    os.makedirs(outdir, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axes[0].plot(sim.tau_hist, np.degrees(sim.a_eff_hist), "b-", lw=1.5)
    axes[0].set_ylabel("α_eff (°)")
    axes[0].set_title(f"Plunging History (kh = {kh_val:.2f})", fontweight="bold")
    axes[0].grid(alpha=0.3)

    axes[1].plot(sim.tau_hist, sim.Cl_hist, "b-", lw=1.5)
    axes[1].set_ylabel("Cₗ")
    axes[1].grid(alpha=0.3)

    axes[2].plot(sim.tau_hist, sim.Ct_hist, "b-", lw=1.5)
    axes[2].axhline(0, color="k", alpha=0.3)
    axes[2].set_xlabel("τ (semi-chords)")
    axes[2].set_ylabel("Cₜ")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{outdir}/history_kh{kh_val:.2f}.png", dpi=150)
    plt.close()


def plot_final_wake(sim, kh_val, outdir):
    os.makedirs(outdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    x_base, y_base = get_naca0012_coords(CHORD, 40)

    wx, wy, wg = np.array(sim.wake_x), np.array(sim.wake_y), np.array(sim.wake_gamma)
    if len(wx) > 0:
        sz = 50 * np.abs(wg) / (np.max(np.abs(wg)) or 1.0) + 2
        ax.scatter(
            wx[wg > 0],
            wy[wg > 0],
            s=sz[wg > 0],
            c="#d62728",
            alpha=0.6,
            edgecolors="none",
            label="Γ>0 (CCW)",
        )
        ax.scatter(
            wx[wg < 0],
            wy[wg < 0],
            s=sz[wg < 0],
            c="#1f77b4",
            alpha=0.6,
            edgecolors="none",
            label="Γ<0 (CW)",
        )

    snap = sim.wake_snapshots[-1]
    ca, sa = np.cos(-snap["alpha"]), np.sin(-snap["alpha"])
    x_rot = sim.x0 + (x_base - sim.x0) * ca - y_base * sa
    y_rot = snap["h"] + (x_base - sim.x0) * sa + y_base * ca

    ax.fill(x_rot, y_rot, color="#333", alpha=0.8)
    ax.plot(sim.x0, snap["h"], "ro", markersize=5, zorder=10, label="Pivot")

    ax.set_title(f"Plunging Wake (kh = {kh_val:.2f})")
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f"{outdir}/wake_kh{kh_val:.2f}.png", dpi=150)
    plt.close()


def create_animation(sim, kh_val, outdir):
    os.makedirs(outdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    x_base, y_base = get_naca0012_coords(CHORD, 50)
    x_max = U_INF * (N_CYCLES / sim.f) * 1.1

    def animate(i):
        ax.clear()
        snap = sim.wake_snapshots[i]
        if len(snap["wx"]) > 0:
            wg = np.array(snap["wg"])
            sizes = 40 * np.abs(wg) / (np.max(np.abs(wg)) + 1e-10) + 10
            colors = ["#d62728" if g > 0 else "#1f77b4" for g in wg]
            ax.scatter(
                snap["wx"], snap["wy"], c=colors, s=sizes, alpha=0.6, edgecolors="none"
            )

        ca, sa = np.cos(-snap["alpha"]), np.sin(-snap["alpha"])
        x_rot = sim.x0 + (x_base - sim.x0) * ca - y_base * sa
        y_rot = snap["h"] + (x_base - sim.x0) * sa + y_base * ca

        ax.add_patch(
            Polygon(
                np.column_stack([x_rot, y_rot]),
                facecolor="#333",
                edgecolor="black",
                alpha=0.9,
            )
        )
        ax.plot(sim.x0, snap["h"], "ro", markersize=5, zorder=10)

        ax.axhline(0, color="gray", linestyle="--", alpha=0.4)
        ax.set_xlim(-0.5, max(3, x_max))
        ax.set_ylim(-max(1.5, sim.h0 * 3), max(1.5, sim.h0 * 3))
        ax.set_aspect("equal")
        ax.set_title(f"Plunging | kh = {kh_val:.2f} | t = {snap['t']:.2f}s")

    anim = animation.FuncAnimation(
        fig, animate, frames=len(sim.wake_snapshots), interval=1000 / FPS
    )
    try:
        anim.save(f"{outdir}/animation.mp4", writer=animation.FFMpegWriter(fps=FPS))
    except Exception:
        anim.save(f"{outdir}/animation.gif", writer=animation.PillowWriter(fps=FPS))
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    kh = k_paper * h_paper
    out_dir = "results/plunge"

    print(f"Running Plunging Simulation (kh = {kh:.2f})...")
    sim = PlungeVortexSolver(k_paper, h_paper, include_added_mass=True)
    sim.simulate(N_CYCLES)

    print("Generating plots and animation...")
    plot_history(sim, kh, out_dir)
    plot_final_wake(sim, kh, out_dir)
    create_animation(sim, kh, out_dir)
    print(f"Done. Outputs saved to ./{out_dir}/")
