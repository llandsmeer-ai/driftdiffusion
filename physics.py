import jax
import jax.numpy as jnp
import chex
from mesh import Grid

@chex.dataclass
class Material:
    mu_n: float       # Scaled electron mobility
    mu_p: float       # Scaled hole mobility
    tau_n: float      # Scaled electron lifetime
    tau_p: float      # Scaled hole lifetime
    N_dop: jnp.ndarray # Net doping N_D - N_A (scaled)

@chex.dataclass
class State:
    psi: jnp.ndarray
    n: jnp.ndarray
    p: jnp.ndarray

def safe_bernoulli(x, eps=1e-6):
    """
    Computes B(x) = x / (exp(x) - 1).
    Uses Taylor expansion for |x| < eps to avoid NaN during AD.
    B(x) approx 1 - x/2 + x^2/12
    """
    safe_x = jnp.where(jnp.abs(x) < eps, eps, x)
    b_val = safe_x / (jnp.exp(safe_x) - 1.0)
    
    # Taylor series near x=0
    taylor_val = 1.0 - x / 2.0 + (x**2) / 12.0
    
    return jnp.where(jnp.abs(x) < eps, taylor_val, b_val)

def compute_srh_recombination(n, p, material):
    """
    Computes scaled SRH recombination rate.
    R = (n*p - 1) / (tau_p * (n + 1) + tau_n * (p + 1))
    Assumes trap energy is at intrinsic level (n_1 = p_1 = 1 in scaled units).
    """
    numerator = n * p - 1.0
    denominator = material.tau_p * (n + 1.0) + material.tau_n * (p + 1.0)
    return numerator / denominator

def compute_residuals(state, old_state, dt, grid, material):
    """
    Computes the residuals for the drift-diffusion equations.
    
    Args:
        state: State at t + dt
        old_state: State at t
        dt: time step (if jnp.inf, steady-state)
        grid: Grid object
        material: Material object
        
    Returns:
        F_psi: Poisson residual
        F_n: Electron continuity residual
        F_p: Hole continuity residual
    """
    psi = state.psi
    n = state.n
    p = state.p
    
    dx = grid.dx
    dx_mid = grid.dx_mid
    
    # 1. Edge Potentials and Electric Field
    dpsi = jnp.diff(psi) # psi_{i+1} - psi_i
    
    # 2. Scharfetter-Gummel Fluxes
    # J_n = mu_n / dx * (n_{i+1} * B(dpsi) - n_i * B(-dpsi))
    # J_p = mu_p / dx * (p_i * B(dpsi) - p_{i+1} * B(-dpsi))
    
    B_pos = safe_bernoulli(dpsi)
    B_neg = safe_bernoulli(-dpsi)
    
    Jn = (material.mu_n / dx) * (n[1:] * B_pos - n[:-1] * B_neg)
    Jp = (material.mu_p / dx) * (p[:-1] * B_pos - p[1:] * B_neg)
    
    # Divergence of fluxes (dJ/dx)
    div_Jn = jnp.diff(Jn) / dx_mid[1:-1]
    div_Jp = jnp.diff(Jp) / dx_mid[1:-1]
    
    # 3. Poisson Residual (Laplacian)
    laplacian_psi = (dpsi[1:] / dx[1:] - dpsi[:-1] / dx[:-1]) / dx_mid[1:-1]
    
    # 4. Recombination (interior nodes)
    R = compute_srh_recombination(n[1:-1], p[1:-1], material)
    
    # 5. Assemble Residuals (Interior Nodes)
    # F_psi = div grad psi - (n - p - N_dop)
    F_psi_int = laplacian_psi - (n[1:-1] - p[1:-1] - material.N_dop[1:-1])
    
    # F_n = (n - n_old) / dt - div Jn + R
    # F_p = (p - p_old) / dt + div Jp + R
    inv_dt = jnp.where(jnp.isinf(dt), 0.0, 1.0 / dt)
    
    F_n_int = inv_dt * (n[1:-1] - old_state.n[1:-1]) - div_Jn + R
    F_p_int = inv_dt * (p[1:-1] - old_state.p[1:-1]) + div_Jp + R
    
    # 6. Boundary Conditions (Initialize with 0, will be overwritten by boundary handler)
    F_psi = jnp.zeros_like(psi).at[1:-1].set(F_psi_int)
    F_n = jnp.zeros_like(n).at[1:-1].set(F_n_int)
    F_p = jnp.zeros_like(p).at[1:-1].set(F_p_int)
    
    return F_psi, F_n, F_p

def compute_currents(state, grid, material):
    """
    Computes the scaled electron and hole current densities at the edges.
    """
    psi = state.psi
    n = state.n
    p = state.p
    dx = grid.dx
    
    dpsi = jnp.diff(psi)
    
    B_pos = safe_bernoulli(dpsi)
    B_neg = safe_bernoulli(-dpsi)
    
    Jn = (material.mu_n / dx) * (n[1:] * B_pos - n[:-1] * B_neg)
    Jp = (material.mu_p / dx) * (p[:-1] * B_pos - p[1:] * B_neg)
    
    return Jn, Jp
