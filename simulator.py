import jax
import jax.numpy as jnp
from physics import State
from solver import solve_newton_step, get_equilibrium_contacts

def get_initial_guess(grid, material, V_applied=(0.0, 0.0)):
    """
    Creates an initial guess based on local charge neutrality and linear potential drop.
    """
    N_dop = material.N_dop
    n = (N_dop + jnp.sqrt(N_dop**2 + 4.0)) / 2.0
    p = 1.0 / n
    psi_builtin = jnp.log(n)
    
    bc_psi, _, _ = get_equilibrium_contacts(N_dop, V_applied)
    
    x_norm = (grid.x - grid.x[0]) / (grid.x[-1] - grid.x[0])
    
    psi = psi_builtin + V_applied[0] + (V_applied[1] - V_applied[0]) * x_norm
    
    return State(psi=psi, n=n, p=p)

def solve_steady_state(grid, material, V_applied=(0.0, 0.0), voltage_steps=10, guess_state=None):
    """
    Solves for steady state by ramping the voltage in steps to maintain convergence.
    """
    print(f"Solving steady-state for V_applied = {V_applied}")
    
    if guess_state is None:
        state = get_initial_guess(grid, material, (0.0, 0.0))
    else:
        state = guess_state
        
    if voltage_steps == 1:
        V_left = jnp.array([V_applied[0]])
        V_right = jnp.array([V_applied[1]])
    else:
        V_left = jnp.linspace(0.0, V_applied[0], voltage_steps)
        V_right = jnp.linspace(0.0, V_applied[1], voltage_steps)
    
    for i in range(voltage_steps):
        V_step = (V_left[i], V_right[i])
        bc_psi, bc_n, bc_p = get_equilibrium_contacts(material.N_dop, V_step)
        
        state, iters, error = solve_newton_step(state, state, jnp.inf, grid, material, bc_psi, bc_n, bc_p)
        print(f"  Step {i+1}/{voltage_steps}, V=({V_step[0]:.3f}, {V_step[1]:.3f}), Iters={iters}, Max Error={error:.2e}")
        
    return state

def run_transient(grid, material, initial_state, dt, n_steps, V_applied=(0.0, 0.0)):
    """
    Runs a transient simulation using Backward Euler time stepping.
    """
    print(f"Running transient for {n_steps} steps with dt={dt}, V_applied={V_applied}")
    bc_psi, bc_n, bc_p = get_equilibrium_contacts(material.N_dop, V_applied)
    
    state = initial_state
    states = [state]
    
    for i in range(n_steps):
        new_state, iters, error = solve_newton_step(state, state, dt, grid, material, bc_psi, bc_n, bc_p)
        state = new_state
        states.append(state)
        
        if (i+1) % 10 == 0 or i == 0 or i == n_steps - 1:
            print(f"  Time Step {i+1}/{n_steps}, Iters={iters}, Max Error={error:.2e}")
            
    return states
