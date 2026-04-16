import jax
import jax.numpy as jnp
from functools import partial
import chex
from physics import State, compute_residuals

def get_equilibrium_contacts(N_dop, V_applied=(0.0, 0.0)):
    """
    Computes Dirichlet boundary values for Ohmic contacts.
    Args:
        N_dop: array of net doping (N_D - N_A)
        V_applied: tuple of (V_left, V_right) applied biases (scaled)
    Returns:
        psi_bc: (left, right)
        n_bc: (left, right)
        p_bc: (left, right)
    """
    n_left = (N_dop[0] + jnp.sqrt(N_dop[0]**2 + 4.0)) / 2.0
    n_right = (N_dop[-1] + jnp.sqrt(N_dop[-1]**2 + 4.0)) / 2.0
    
    p_left = 1.0 / n_left
    p_right = 1.0 / n_right
    
    psi_builtin_left = jnp.log(n_left)
    psi_builtin_right = jnp.log(n_right)
    
    psi_left = psi_builtin_left + V_applied[0]
    psi_right = psi_builtin_right + V_applied[1]
    
    return (psi_left, psi_right), (n_left, n_right), (p_left, p_right)

def apply_boundary_residuals(F_psi, F_n, F_p, state, bc_psi, bc_n, bc_p):
    F_psi = F_psi.at[0].set(state.psi[0] - bc_psi[0])
    F_psi = F_psi.at[-1].set(state.psi[-1] - bc_psi[1])
    F_n = F_n.at[0].set(state.n[0] - bc_n[0])
    F_n = F_n.at[-1].set(state.n[-1] - bc_n[1])
    F_p = F_p.at[0].set(state.p[0] - bc_p[0])
    F_p = F_p.at[-1].set(state.p[-1] - bc_p[1])
    return F_psi, F_n, F_p

def flatten_state(state):
    """Flattens State into a single vector [psi_0, n_0, p_0, psi_1, n_1, p_1, ...]"""
    return jnp.stack([state.psi, state.n, state.p], axis=-1).flatten()

def unflatten_state(X, N):
    """Unflattens vector into State"""
    X_reshaped = X.reshape((N, 3))
    return State(psi=X_reshaped[:, 0], n=X_reshaped[:, 1], p=X_reshaped[:, 2])

def unflatten_residuals(F, N):
    F_reshaped = F.reshape((N, 3))
    return F_reshaped[:, 0], F_reshaped[:, 1], F_reshaped[:, 2]

def full_residual_fn(X_flat, old_state, dt, grid, material, bc_psi, bc_n, bc_p):
    """Wrapper function to compute the flattened residual vector for Newton."""
    N = len(grid.x)
    state = unflatten_state(X_flat, N)
    F_psi, F_n, F_p = compute_residuals(state, old_state, dt, grid, material)
    F_psi, F_n, F_p = apply_boundary_residuals(F_psi, F_n, F_p, state, bc_psi, bc_n, bc_p)
    return jnp.stack([F_psi, F_n, F_p], axis=-1).flatten()

@partial(jax.jit, static_argnames=['max_iters'])
def solve_newton_step(state, old_state, dt, grid, material, bc_psi, bc_n, bc_p, max_iters=100, tol=1e-8):
    """
    JIT-compiled Newton-Raphson solver for a single time step or steady-state.
    """
    N = len(grid.x)
    X_flat = flatten_state(state)
    
    # We use a custom jax.lax.while_loop to keep it entirely compiled on GPU/CPU
    def cond_fn(loop_state):
        i, X, error = loop_state
        return (i < max_iters) & (error > tol)
        
    def body_fn(loop_state):
        i, X, _ = loop_state
        
        # Calculate F(X)
        F = full_residual_fn(X, old_state, dt, grid, material, bc_psi, bc_n, bc_p)
        
        # Calculate Jacobian J = dF/dX
        J = jax.jacfwd(full_residual_fn, argnums=0)(X, old_state, dt, grid, material, bc_psi, bc_n, bc_p)
        
        # Solve J * dX = -F
        # For 1D, jnp.linalg.solve is perfectly fine and fast for N < 1000
        dX = jnp.linalg.solve(J, -F)
        
        # Damping: limit max potential update to ~2 V_T (which is 2 in scaled units)
        # We need to unflatten dX, clip dpsi, then reflatten.
        dX_reshaped = dX.reshape((N, 3))
        dpsi = dX_reshaped[:, 0]
        
        # Simple damping for potential to avoid huge swings
        dpsi_max = jnp.max(jnp.abs(dpsi))
        damping = jnp.where(dpsi_max > 2.0, 2.0 / dpsi_max, 1.0)
        
        dX_damped = dX * damping
        
        X_new = X + dX_damped
        
        # Also ensure n and p don't go negative (if they do, clamp to a small positive number)
        # This is a rough hack, better is to use quasi-Fermi potentials, but this works for simple cases
        state_new = unflatten_state(X_new, N)
        n_safe = jnp.where(state_new.n < 1e-30, 1e-30, state_new.n)
        p_safe = jnp.where(state_new.p < 1e-30, 1e-30, state_new.p)
        X_new_clamped = flatten_state(State(psi=state_new.psi, n=n_safe, p=p_safe))
        
        error = jnp.linalg.norm(F, ord=jnp.inf)
        
        return (i + 1, X_new_clamped, error)
        
    # Initial error
    initial_F = full_residual_fn(X_flat, old_state, dt, grid, material, bc_psi, bc_n, bc_p)
    initial_error = jnp.linalg.norm(initial_F, ord=jnp.inf)
    
    init_loop_state = (0, X_flat, initial_error)
    
    final_loop_state = jax.lax.while_loop(cond_fn, body_fn, init_loop_state)
    
    iters, X_final, final_error = final_loop_state
    
    return unflatten_state(X_final, N), iters, final_error
