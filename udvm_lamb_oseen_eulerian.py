"""
=============================================================================
Method: Unsteady Discrete Vortex Method (UDVM)
Motion: Plunging Only
Geometry: NACA 4-Digit Camber Line
=============================================================================
Description:
A Numba-accelerated UDVM solver utilizing a Lamb-Oseen viscous core model to
desingularize vortex interactions, preventing numerical blowups. It projects the
wake onto an Eulerian grid for fluid-like visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import njit, prange
import time

# --- Numba Accelerated Kernels (Upgraded with Lamb-Oseen Model) ---


@njit(parallel=True, fastmath=True)
def compute_induced_velocity_numba(
    target_x, target_y, source_x, source_y, Gamma, sigma, eps
):
    """Biot-Savart kernel using Lamb-Oseen viscous core for desingularization."""
    M = target_x.shape[0]
    N = source_x.shape[0]
    u = np.zeros(M)
    v = np.zeros(M)

    inv_2pi = 1.0 / (2.0 * np.pi)

    for i in prange(M):
        tx = target_x[i]
        ty = target_y[i]
        ui = 0.0
        vi = 0.0
        for j in range(N):
            dx = tx - source_x[j]
            dy = ty - source_y[j]
            r2 = dx * dx + dy * dy + eps

            if r2 < eps * 2:
                continue

            sig2 = sigma[j] ** 2
            # Lamb-Oseen factor
            factor = (Gamma[j] * inv_2pi / r2) * (1.0 - np.exp(-r2 / sig2))

            ui += factor * dy
            vi -= factor * dx
        u[i] = ui
        v[i] = vi
    return u, v


@njit(parallel=True, fastmath=True)
def convect_wake_numba(wx, wy, wG, wS, vx, vy, vG, bS, U_inf, dt, eps):
    """Calculates induced velocities and convects the wake."""
    # 1. Wake-on-Wake influence
    uw, vw = compute_induced_velocity_numba(wx, wy, wx, wy, wG, wS, eps)

    # 2. Bound-on-Wake influence
    ub, vb = compute_induced_velocity_numba(wx, wy, vx, vy, vG, bS, eps)

    # 3. Update positions and grow core size (viscous diffusion)
    new_wx = wx + (U_inf + uw + ub) * dt
    new_wy = wy + (vw + vb) * dt
    new_wS = np.sqrt(wS**2 + 4 * 1e-4 * dt)  # Artificial kinematic viscosity

    return new_wx, new_wy, new_wS


@njit(parallel=True, fastmath=True)
def render_omega_field(X_grid, Y_grid, wp_x, wp_y, wp_gamma, wp_sigma, x_lim_max):
    """Projects discrete Lagrangian vortices onto an Eulerian grid for visualization."""
    rows, cols = X_grid.shape
    omega_field = np.zeros((rows, cols))
    num_particles = wp_x.shape[0]

    for i in prange(rows):
        for j in range(cols):
            val = 0.0
            xg = X_grid[i, j]
            yg = Y_grid[i, j]
            for k in range(num_particles):
                # Optimization: Skip particles far downstream of the render window
                if wp_x[k] > x_lim_max + 1.0:
                    continue
                r2_grid = (xg - wp_x[k]) ** 2 + (yg - wp_y[k]) ** 2
                sig2 = wp_sigma[k] ** 2
                # Only compute for grid points within 4 standard deviations
                if r2_grid < 16 * sig2:
                    val += wp_gamma[k] * (1 / (np.pi * sig2)) * np.exp(-r2_grid / sig2)
            omega_field[i, j] = val
    return omega_field


# --- UDVM Physics Engine ---


class PlungingAirfoilUDVM:
    def __init__(self, m, p, c, N, U_inf, k, h_amp, dt, sigma_core=0.05):
        self.c, self.N, self.U_inf, self.dt = c, N, U_inf, dt
        self.omega = 2 * k * U_inf / c
        self.h0 = h_amp * c
        self.eps = 1e-8
        self.sigma_core = sigma_core

        # Thin airfoil camber line discretization
        self.dx = c / N
        self.x_nodes = np.linspace(0, c, N + 1)
        self.x_vortex = self.x_nodes[:-1] + 0.25 * self.dx
        self.x_colloc = self.x_nodes[:-1] + 0.75 * self.dx

        self.y_vortex_base = self._camber_line(self.x_vortex, m, p, c)
        self.y_colloc_base = self._camber_line(self.x_colloc, m, p, c)
        self.normals = self._camber_normals(self.x_colloc, m, p, c)

        # State arrays
        self.wake_x = np.array([], dtype=np.float64)
        self.wake_y = np.array([], dtype=np.float64)
        self.wake_Gamma = np.array([], dtype=np.float64)
        self.wake_Sigma = np.array([], dtype=np.float64)
        self.Gamma_bound = np.zeros(N, dtype=np.float64)
        self.bound_Sigma = np.full(N, sigma_core, dtype=np.float64)

        # Pre-compute rigid aerodynamic influence matrix
        dx = self.x_colloc[:, np.newaxis] - self.x_vortex[np.newaxis, :]
        dy = self.y_colloc_base[:, np.newaxis] - self.y_vortex_base[np.newaxis, :]
        r2 = dx**2 + dy**2 + self.eps
        u = (1.0 / (2 * np.pi)) * (dy / r2)
        v = -(1.0 / (2 * np.pi)) * (dx / r2)
        self.A = u * self.normals[:, 0:1] + v * self.normals[:, 1:2]

    def _camber_line(self, x, m, p, c):
        if m == 0 or p == 0:
            return np.zeros_like(x)
        return np.where(
            x < p * c,
            m * (x / p**2) * (2 * p - x / c),
            m * ((c - x) / (1 - p) ** 2) * (1 + x / c - 2 * p),
        )

    def _camber_normals(self, x, m, p, c):
        if m == 0 or p == 0:
            return np.tile(np.array([0.0, 1.0]), (len(x), 1))
        dy_dx = np.where(
            x < p * c, 2 * m / p**2 * (p - x / c), 2 * m / (1 - p) ** 2 * (p - x / c)
        )
        n = np.column_stack((-dy_dx, np.ones_like(x)))
        return n / np.linalg.norm(n, axis=1)[:, np.newaxis]

    def step(self, t):
        # Kinematics
        h_t = self.h0 * np.sin(self.omega * t)
        dh_dt = self.h0 * self.omega * np.cos(self.omega * t)
        y_v_curr = self.y_vortex_base + h_t
        y_c_curr = self.y_colloc_base + h_t

        # Shedding Location (0.3 * U_inf * dt downstream of TE)
        new_wx, new_wy = self.c + 0.3 * self.U_inf * self.dt, y_v_curr[-1]

        # Induced velocity from wake and newly shed vortex on collocation points
        uw, vw = compute_induced_velocity_numba(
            self.x_colloc,
            y_c_curr,
            self.wake_x,
            self.wake_y,
            self.wake_Gamma,
            self.wake_Sigma,
            self.eps,
        )
        un, vn = compute_induced_velocity_numba(
            self.x_colloc,
            y_c_curr,
            np.array([new_wx]),
            np.array([new_wy]),
            np.array([1.0]),
            np.array([self.sigma_core]),
            self.eps,
        )

        # Build (N+1) x (N+1) system
        sys_A = np.zeros((self.N + 1, self.N + 1))
        sys_A[: self.N, : self.N] = self.A
        sys_A[: self.N, self.N] = un * self.normals[:, 0] + vn * self.normals[:, 1]
        sys_A[self.N, : self.N] = 1.0  # Kelvin's Theorem: Sum of new bound
        sys_A[self.N, self.N] = 1.0  # Kelvin's Theorem: + new shed

        sys_b = np.zeros(self.N + 1)
        sys_b[: self.N] = -(
            (self.U_inf + uw) * self.normals[:, 0] + (dh_dt + vw) * self.normals[:, 1]
        )
        sys_b[self.N] = np.sum(self.Gamma_bound)  # Kelvin's Theorem: = Sum of old bound

        # Solve for bound circulation and new shed circulation
        sol = np.linalg.solve(sys_A, sys_b)
        self.Gamma_bound, new_wG = sol[: self.N], sol[self.N]

        # Convect existing wake
        if self.wake_x.size > 0:
            self.wake_x, self.wake_y, self.wake_Sigma = convect_wake_numba(
                self.wake_x,
                self.wake_y,
                self.wake_Gamma,
                self.wake_Sigma,
                self.x_vortex,
                y_v_curr,
                self.Gamma_bound,
                self.bound_Sigma,
                self.U_inf,
                self.dt,
                self.eps,
            )

        # Append newly shed vortex
        self.wake_x = np.append(self.wake_x, new_wx)
        self.wake_y = np.append(self.wake_y, new_wy)
        self.wake_Gamma = np.append(self.wake_Gamma, new_wG)
        self.wake_Sigma = np.append(self.wake_Sigma, self.sigma_core)

        return self.x_nodes, y_v_curr


# --- Execution and Animation ---
if __name__ == "__main__":
    # Parameters
    k_reduced = 1
    h_amp = 0.6
    k_h = k_reduced * h_amp
    dt = 0.05
    steps = 1200

    print(f"Initializing Unified Solver for kh = {k_h:.3f}")
    model = PlungingAirfoilUDVM(
        m=0.02,
        p=0.4,
        c=1.0,
        N=30,
        U_inf=1.0,
        k=k_reduced,
        h_amp=h_amp,
        dt=dt,
        sigma_core=0.1,
    )

    # Rendering Setup
    grid_res_x, grid_res_y = 250, 100
    x_lim_min, x_lim_max = -0.5, 25.0
    y_lim_min, y_lim_max = -3.0, 3.0
    X_grid, Y_grid = np.meshgrid(
        np.linspace(x_lim_min, x_lim_max, grid_res_x),
        np.linspace(y_lim_min, y_lim_max, grid_res_y),
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    img = ax.imshow(
        np.zeros_like(X_grid),
        extent=[x_lim_min, x_lim_max, y_lim_min, y_lim_max],
        origin="lower",
        cmap="RdBu_r",
        vmin=-5,
        vmax=5,
        interpolation="bicubic",
    )

    (airfoil_line,) = ax.plot([], [], "k-", lw=3)
    ax.set_xlim([x_lim_min, x_lim_max])
    ax.set_ylim([y_lim_min, y_lim_max])
    ax.set_aspect("equal")
    ax.set_title(f"Coupled UDVM Wake Visualization (kh = {k_h:.2f})")

    def animate(frame):
        t = frame * dt
        x_n, y_n = model.step(t)

        # Render Airfoil
        airfoil_line.set_data(x_n, np.interp(x_n, model.x_vortex, y_n))

        # Render Eulerian Field
        if model.wake_x.size > 0:
            omega_field = render_omega_field(
                X_grid,
                Y_grid,
                model.wake_x,
                model.wake_y,
                model.wake_Gamma,
                model.wake_Sigma,
                x_lim_max,
            )
            img.set_data(omega_field)

        return img, airfoil_line

    print(f"Generating animation ({steps} frames)...")
    start_time = time.time()

    ani = animation.FuncAnimation(fig, animate, frames=steps, blit=True, interval=20)

    # Save video (Ensure you have ffmpeg installed, or change to .gif / pillow writer)
    filename = "udvm_lamb_oseen_eulerian.mp4"
    ani.save(filename, writer="ffmpeg", fps=20)

    print(f"Animation saved to {filename} in {time.time() - start_time:.2f} seconds.")
