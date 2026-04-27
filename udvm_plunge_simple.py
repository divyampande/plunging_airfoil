# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation

# class PlungingAirfoilUDVM:
#     def __init__(self, m, p, c, N, U_inf, k, h_amp, dt):
#         """
#         m, p: NACA 4-digit camber parameters (e.g., 0,0 for symmetric)
#         c: Chord length
#         N: Number of panels
#         U_inf: Freestream velocity
#         k: Reduced frequency (k = omega * c / (2 * U_inf))
#         h_amp: Non-dimensional plunge amplitude (h0 / c)
#         dt: Time step
#         """
#         self.c = c
#         self.N = N
#         self.U_inf = U_inf
#         self.omega = 2 * k * U_inf / c
#         self.h0 = h_amp * c
#         self.dt = dt

#         # Discretize camber line
#         self.dx = c / N
#         self.x_nodes = np.linspace(0, c, N + 1)
#         self.x_vortex = self.x_nodes[:-1] + 0.25 * self.dx  # 1/4 chord
#         self.x_colloc = self.x_nodes[:-1] + 0.75 * self.dx  # 3/4 chord

#         # Calculate camber line y-coordinates and normal vectors
#         self.y_vortex = self._camber_line(self.x_vortex, m, p, c)
#         self.y_colloc = self._camber_line(self.x_colloc, m, p, c)
#         self.normals = self._camber_normals(self.x_colloc, m, p, c)

#         # Initialize wake and circulation arrays
#         self.wake_x = []
#         self.wake_y = []
#         self.wake_Gamma = []
#         self.Gamma_bound = np.zeros(N)

#         # Pre-compute rigid influence matrix [N x N]
#         self.A = np.zeros((N, N))
#         for i in range(N):
#             for j in range(N):
#                 u, v = self._biot_savart(self.x_colloc[i], self.y_colloc[i],
#                                          self.x_vortex[j], self.y_vortex[j], 1.0)
#                 self.A[i, j] = u * self.normals[i, 0] + v * self.normals[i, 1]

#     def _camber_line(self, x, m, p, c):
#         if m == 0 or p == 0:
#             return np.zeros_like(x)
#         y = np.where(x < p * c,
#                      m * (x / p**2) * (2 * p - x / c),
#                      m * ((c - x) / (1 - p)**2) * (1 + x / c - 2 * p))
#         return y

#     def _camber_normals(self, x, m, p, c):
#         # Simplified for small camber: normal is roughly [-dy/dx, 1] normalized
#         if m == 0 or p == 0:
#             return np.array([[0, 1] for _ in x])

#         dy_dx = np.where(x < p * c,
#                          2 * m / p**2 * (p - x / c),
#                          2 * m / (1 - p)**2 * (p - x / c))

#         normals = np.column_stack((-dy_dx, np.ones_like(x)))
#         return normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]

#     def _biot_savart(self, x, y, xv, yv, Gamma):
#         # Calculates induced velocity (u, v) at (x,y) due to vortex at (xv, yv)
#         dx = x - xv
#         dy = y - yv
#         r2 = dx**2 + dy**2 + 1e-10 # core radius to prevent singularity
#         u =  (Gamma / (2 * np.pi)) * (dy / r2)
#         v = -(Gamma / (2 * np.pi)) * (dx / r2)
#         return u, v

#     def step(self, t):
#         # Kinematics of plunging
#         h_t = self.h0 * np.sin(self.omega * t)
#         dh_dt = self.h0 * self.omega * np.cos(self.omega * t)

#         # Update bound vortex and collocation y-positions based on plunge
#         y_v_current = self.y_vortex + h_t
#         y_c_current = self.y_colloc + h_t

#         # Setup (N+1) x (N+1) system to solve for new bound Gammas and the new shed wake Gamma
#         # Row 0 to N-1: Zero penetration boundary condition
#         # Row N: Kelvin's circulation theorem
#         sys_A = np.zeros((self.N + 1, self.N + 1))
#         sys_A[:self.N, :self.N] = self.A

#         sys_b = np.zeros(self.N + 1)

#         # Position of newly shed vortex (0.3 * U_inf * dt downstream of TE)
#         new_wake_x = self.c + 0.3 * self.U_inf * self.dt
#         new_wake_y = y_v_current[-1]

#         # Populate RHS and Wake influence
#         for i in range(self.N):
#             # Kinematic velocity at collocation point
#             V_kin_x = self.U_inf
#             V_kin_y = dh_dt

#             # Induced velocity from existing wake
#             u_ind_wake, v_ind_wake = 0.0, 0.0
#             for wx, wy, wG in zip(self.wake_x, self.wake_y, self.wake_Gamma):
#                 u, v = self._biot_savart(self.x_colloc[i], y_c_current[i], wx, wy, wG)
#                 u_ind_wake += u
#                 v_ind_wake += v

#             # Influence of the newly shedding vortex on collocation points
#             u_new_w, v_new_w = self._biot_savart(self.x_colloc[i], y_c_current[i], new_wake_x, new_wake_y, 1.0)
#             sys_A[i, self.N] = u_new_w * self.normals[i, 0] + v_new_w * self.normals[i, 1]

#             # RHS: -(V_kin + V_ind_wake) dot n
#             sys_b[i] = -((V_kin_x + u_ind_wake) * self.normals[i, 0] +
#                          (V_kin_y + v_ind_wake) * self.normals[i, 1])

#         # Kelvin's Theorem: sum(Gamma_bound_new) + Gamma_new_wake = sum(Gamma_bound_old)
#         sys_A[self.N, :self.N] = 1.0
#         sys_A[self.N, self.N] = 1.0
#         sys_b[self.N] = np.sum(self.Gamma_bound)

#         # Solve system
#         solution = np.linalg.solve(sys_A, sys_b)
#         self.Gamma_bound = solution[:self.N]
#         new_wake_Gamma = solution[self.N]

#         # Convect existing wake vortices using 1st order Euler (can be upgraded to RK4)
#         for i in range(len(self.wake_x)):
#             u_total = self.U_inf
#             v_total = 0.0

#             # Influence from bound vortices
#             for j in range(self.N):
#                 u, v = self._biot_savart(self.wake_x[i], self.wake_y[i],
#                                          self.x_vortex[j], y_v_current[j], self.Gamma_bound[j])
#                 u_total += u
#                 v_total += v

#             # Influence from other wake vortices
#             for j in range(len(self.wake_x)):
#                 if i != j:
#                     u, v = self._biot_savart(self.wake_x[i], self.wake_y[i],
#                                              self.wake_x[j], self.wake_y[j], self.wake_Gamma[j])
#                     u_total += u
#                     v_total += v

#             self.wake_x[i] += u_total * self.dt
#             self.wake_y[i] += v_total * self.dt

#         # Append newly shed vortex to the wake arrays
#         self.wake_x.append(new_wake_x)
#         self.wake_y.append(new_wake_y)
#         self.wake_Gamma.append(new_wake_Gamma)

#         return self.x_nodes, y_v_current

# # --- Simulation and Animation Parameters ---
# if __name__ == "__main__":
#     c = 1.0
#     U_inf = 1.0
#     dt = 0.05
#     steps = 1000

#     # User Parameters
#     k = 1.5       # Reduced frequency
#     h_amp = 0.1  # Non-dimensional plunge amplitude (h0/c)
#     kh = k * h_amp # Non-dimensional plunge velocity metric
#     print(f"Simulating for kh = {kh:.3f}")

#     # Initialize Model (NACA 0012 effectively, symmetric flat plate)
#     model = PlungingAirfoilUDVM(m=0, p=0, c=c, N=20, U_inf=U_inf, k=k, h_amp=h_amp, dt=dt)

#     fig, ax = plt.subplots(figsize=(10, 4))
#     ax.set_xlim(-0.5, 20.0)
#     ax.set_ylim(-5.0, 5.0)
#     ax.set_aspect('equal')
#     ax.set_title(f'Plunging Airfoil Wake ($k_h = {kh:.2f}$)')
#     ax.set_xlabel('x/c')
#     ax.set_ylabel('y/c')
#     ax.grid(True, linestyle='--', alpha=0.6)

#     airfoil_line, = ax.plot([], [], 'k-', lw=3, label='Airfoil')
#     wake_scatter = ax.scatter([], [], c=[], cmap='coolwarm', s=20, vmin=-0.5, vmax=0.5, edgecolor='k', lw=0.5)

#     def init():
#         airfoil_line.set_data([], [])
#         wake_scatter.set_offsets(np.empty((0, 2)))
#         return airfoil_line, wake_scatter

#     def animate(frame):
#         t = frame * dt
#         x_nodes, y_nodes = model.step(t)

#         # Interpolate nodes for smoother line
#         airfoil_line.set_data(x_nodes, np.interp(x_nodes, model.x_vortex, y_nodes))

#         # Update wake
#         if len(model.wake_x) > 0:
#             offsets = np.column_stack((model.wake_x, model.wake_y))
#             wake_scatter.set_offsets(offsets)
#             wake_scatter.set_array(np.array(model.wake_Gamma))

#         return airfoil_line, wake_scatter

#     ani = animation.FuncAnimation(fig, animate, frames=steps, init_func=init, blit=False, interval=50)
#     # plt.show()
#     ani.save(f'plunging_wake_{kh:.2f}.mp4', writer='ffmpeg', fps=20)

"""
=============================================================================
Method: Unsteady Discrete Vortex Method (UDVM)
Motion: Plunging Only
Geometry: NACA 4-Digit Camber Line
=============================================================================
Description:
A stripped-down, standard UDVM solver. Contains older commented-out object
classes alongside an active Numba-accelerated class. It uses basic scatter
plots for the wake rather than complex Eulerian projections.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import njit, prange

# --- Numba Accelerated Kernels ---


@njit(parallel=True, fastmath=True)
def compute_induced_velocity_numba(target_x, target_y, source_x, source_y, Gamma, eps):
    """Highly optimized Biot-Savart kernel using parallel loops."""
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

            # Skip self-induction or exact overlap
            if r2 < eps * 2:
                continue

            pref = Gamma[j] * inv_2pi / r2
            ui += pref * dy
            vi -= pref * dx
        u[i] = ui
        v[i] = vi
    return u, v


@njit(parallel=True, fastmath=True)
def convect_wake_numba(wx, wy, wG, vx, vy, vG, U_inf, dt, eps):
    """Combines all influences to update wake positions in one pass."""
    M = wx.shape[0]
    N_bound = vx.shape[0]

    # 1. Wake-on-Wake influence
    uw, vw = compute_induced_velocity_numba(wx, wy, wx, wy, wG, eps)

    # 2. Bound-on-Wake influence
    ub, vb = compute_induced_velocity_numba(wx, wy, vx, vy, vG, eps)

    # 3. Update positions
    new_wx = wx + (U_inf + uw + ub) * dt
    new_wy = wy + (vw + vb) * dt

    return new_wx, new_wy


# --- Optimized Class ---


class PlungingAirfoilUDVM:
    def __init__(self, m, p, c, N, U_inf, k, h_amp, dt):
        self.c, self.N, self.U_inf, self.dt = c, N, U_inf, dt
        self.omega = 2 * k * U_inf / c
        self.h0 = h_amp * c
        self.eps = 1e-8

        self.dx = c / N
        self.x_nodes = np.linspace(0, c, N + 1)
        self.x_vortex = self.x_nodes[:-1] + 0.25 * self.dx
        self.x_colloc = self.x_nodes[:-1] + 0.75 * self.dx

        self.y_vortex_base = self._camber_line(self.x_vortex, m, p, c)
        self.y_colloc_base = self._camber_line(self.x_colloc, m, p, c)
        self.normals = self._camber_normals(self.x_colloc, m, p, c)

        self.wake_x = np.array([], dtype=np.float64)
        self.wake_y = np.array([], dtype=np.float64)
        self.wake_Gamma = np.array([], dtype=np.float64)
        self.Gamma_bound = np.zeros(N, dtype=np.float64)

        # Pre-compute rigid influence matrix
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
        h_t = self.h0 * np.sin(self.omega * t)
        dh_dt = self.h0 * self.omega * np.cos(self.omega * t)
        y_v_curr = self.y_vortex_base + h_t
        y_c_curr = self.y_colloc_base + h_t

        new_wx, new_wy = self.c + 0.3 * self.U_inf * self.dt, y_v_curr[-1]

        # Use Numba for induced velocity calculations
        uw, vw = compute_induced_velocity_numba(
            self.x_colloc, y_c_curr, self.wake_x, self.wake_y, self.wake_Gamma, self.eps
        )
        un, vn = compute_induced_velocity_numba(
            self.x_colloc,
            y_c_curr,
            np.array([new_wx]),
            np.array([new_wy]),
            np.array([1.0]),
            self.eps,
        )

        # Solve system (Standard NumPy Solve is fast enough for NxN where N is small)
        sys_A = np.zeros((self.N + 1, self.N + 1))
        sys_A[: self.N, : self.N] = self.A
        sys_A[: self.N, self.N] = un * self.normals[:, 0] + vn * self.normals[:, 1]
        sys_A[self.N, :] = 1.0

        sys_b = np.zeros(self.N + 1)
        sys_b[: self.N] = -(
            (self.U_inf + uw) * self.normals[:, 0] + (dh_dt + vw) * self.normals[:, 1]
        )
        sys_b[self.N] = np.sum(self.Gamma_bound)

        sol = np.linalg.solve(sys_A, sys_b)
        self.Gamma_bound, new_wG = sol[: self.N], sol[self.N]

        # Convect Wake using Parallel Numba Kernel
        if self.wake_x.size > 0:
            self.wake_x, self.wake_y = convect_wake_numba(
                self.wake_x,
                self.wake_y,
                self.wake_Gamma,
                self.x_vortex,
                y_v_curr,
                self.Gamma_bound,
                self.U_inf,
                self.dt,
                self.eps,
            )

        self.wake_x = np.append(self.wake_x, new_wx)
        self.wake_y = np.append(self.wake_y, new_wy)
        self.wake_Gamma = np.append(self.wake_Gamma, new_wG)

        return self.x_nodes, y_v_curr


# --- Animation ---
if __name__ == "__main__":
    model = PlungingAirfoilUDVM(0, 0, 1.0, 20, 1, 1.5, 0.4, 0.05)
    kh = 0.41
    print(f"Simulating for kh = {kh:.3f}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(-0.5, 20.0)
    ax.set_ylim(-2.0, 2.0)
    ax.set_aspect("equal")
    (airfoil_line,) = ax.plot([], [], "k-", lw=2)
    wake_scatter = ax.scatter([], [], c=[], cmap="coolwarm", s=8, vmin=-0.5, vmax=0.5)

    def animate(frame):
        x_n, y_n = model.step(frame * 0.05)
        airfoil_line.set_data(x_n, np.interp(x_n, model.x_vortex, y_n))
        if model.wake_x.size > 0:
            wake_scatter.set_offsets(np.column_stack((model.wake_x, model.wake_y)))
            wake_scatter.set_array(model.wake_Gamma)
        return airfoil_line, wake_scatter

    ani = animation.FuncAnimation(fig, animate, frames=1000, blit=True, interval=1)
    # plt.show()
    ani.save("udvm_plunge_simple.mp4", writer="ffmpeg", fps=20)
