import os
import jax
jax.config.update("jax_enable_x64", True) # MUST DO THIS FIRST
import jax.numpy as jnp
import matplotlib.pyplot as plt

from constants import DeMariScaling
from mesh import generate_nonuniform_mesh
from physics import Material, compute_currents
from simulator import solve_steady_state, run_transient
from plot import plot_device_state, plot_iv_curve

def main():
    # 1. Physics Setup & Scaling
    # Silicon at 300K
    scaling = DeMariScaling(T=300.0, ni=1.0e16, eps_r=11.7)
    
    # Device Dimensions: 10 um total length, junction at 5 um
    L_si = 10e-6
    junction_si = 5e-6
    L_dl = scaling.scale_x(L_si)
    junction_dl = scaling.scale_x(junction_si)
    
    print(f"L_dl = {L_dl:.2f}, L_D = {scaling.L_D:.2e} m")
    
    # 2. Mesh Generation
    # 1D Non-uniform mesh with 200 points clustered at the junction
    grid = generate_nonuniform_mesh(L_dl, n_points=200, junction_pos=junction_dl, refinement_factor=3.0)
    
    # 3. Material Properties
    # Mobilities: mu_n = 1000 cm^2/Vs = 0.1 m^2/Vs, mu_p = 500 cm^2/Vs = 0.05 m^2/Vs
    # Lifetimes: tau_n = 1e-6 s, tau_p = 1e-6 s
    
    mu_n_si = 0.1
    mu_p_si = 0.05
    tau_n_si = 1e-6
    tau_p_si = 1e-6
    
    D_n_si = mu_n_si * scaling.V_T
    tau_c = scaling.L_D**2 / D_n_si
    
    print(f"Time scale tau_c = {tau_c:.2e} s")
    
    mu_n_dl = 1.0
    mu_p_dl = mu_p_si / mu_n_si
    tau_n_dl = tau_n_si / tau_c
    tau_p_dl = tau_p_si / tau_c
    
    # Doping: N_D = 1e21 m^-3 (n-type on left), N_A = 1e21 m^-3 (p-type on right)
    N_D_si = 1e21
    N_A_si = 1e21
    
    N_dop_si = jnp.where(grid.x < junction_dl, N_D_si, -N_A_si)
    N_dop_dl = scaling.scale_C(N_dop_si)
    
    material = Material(
        mu_n=mu_n_dl,
        mu_p=mu_p_dl,
        tau_n=tau_n_dl,
        tau_p=tau_p_dl,
        N_dop=N_dop_dl
    )
    
    # 4. Steady-State Equilibrium (0V)
    print("\n--- Computing Thermal Equilibrium ---")
    state_eq = solve_steady_state(grid, material, V_applied=(0.0, 0.0), voltage_steps=1)
    plot_device_state(grid, state_eq, material, scaling, title="Thermal Equilibrium (0V)", filename="eq_state.png")
    
    # # 5. Steady-State Forward Bias (0.5V)
    # V_applied_si = 0.5
    # V_applied_dl = scaling.scale_V(V_applied_si)
    # print(f"\n--- Computing Steady-State Forward Bias ({V_applied_si}V) ---")
    # state_fb = solve_steady_state(grid, material, V_applied=(V_applied_dl, 0.0), voltage_steps=10, guess_state=state_eq)
    # plot_device_state(grid, state_fb, material, scaling, title=f"Forward Bias ({V_applied_si}V)", filename="fb_state.png")
    
    # # 6. Transient (0V -> 0.5V pulse)
    # print(f"\n--- Computing Transient Pulse (0V -> {V_applied_si}V) ---")
    # dt_si = 1e-9 # 1 ns steps
    # dt_dl = dt_si / tau_c
    # n_steps = 100
    # states_transient = run_transient(grid, material, state_eq, dt_dl, n_steps, V_applied=(V_applied_dl, 0.0))
    # plot_device_state(grid, states_transient[-1], material, scaling, title=f"Transient Final State (t={n_steps*dt_si*1e9:.0f} ns)", filename="transient_state.png")
    
    # 7. I-V Curve Sweep (-2.0V to 0.8V)
    print("\n--- Computing I-V Characteristic ---")
    voltages_si = jnp.linspace(-2.0, 0.8, 30)
    currents_si = []
    
    state_sweep = state_eq # Start from equilibrium
    
    for v_si in voltages_si:
        v_dl = scaling.scale_V(v_si)
        # Solve steady state using previous step as guess
        state_sweep = solve_steady_state(grid, material, V_applied=(0.0, v_dl), voltage_steps=10 if v_si > 0.4 else 1, guess_state=state_sweep)
        
        # Calculate terminal current
        Jn, Jp = compute_currents(state_sweep, grid, material)
        
        # Total current at the right terminal
        J_total_dl = Jn[-1] + Jp[-1]
        
        # Scale back to A/m^2, then to A/cm^2
        J_total_si = scaling.unscale_J(J_total_dl)
        J_total_cm2 = J_total_si / 10000.0 # 1 m^2 = 10000 cm^2
        
        currents_si.append(J_total_cm2)
        print(f"  V = {v_si:.2f} V, J = {J_total_cm2:.2e} A/cm^2")
        
    plot_iv_curve(voltages_si, currents_si, title="I-V Characteristic", filename="iv_curve.png")
    
    print("\nSimulation Complete!")

if __name__ == "__main__":
    main()
