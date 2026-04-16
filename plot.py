import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

def plot_device_state(grid, state, material, scaling, title="Device State", filename=None):
    """
    Plots the potential, energy bands, and carrier concentrations.
    """
    # Convert Jax arrays to numpy for plotting
    x_um = np.array(scaling.unscale_x(grid.x) * 1e6) # in micrometers
    psi = np.array(scaling.unscale_V(state.psi))
    n = np.array(scaling.unscale_C(state.n))
    p = np.array(scaling.unscale_C(state.p))
    N_dop = np.array(scaling.unscale_C(material.N_dop))
    
    # Calculate quasi-Fermi levels (scaled first, then unscaled)
    # n = n_i * exp(psi - phi_n) => phi_n = psi - ln(n/n_i)
    # p = n_i * exp(phi_p - psi) => phi_p = psi + ln(p/n_i)
    
    phi_n = state.psi - jnp.log(jnp.maximum(state.n, 1e-30))
    phi_p = state.psi + jnp.log(jnp.maximum(state.p, 1e-30))
    
    phi_n_unscaled = np.array(scaling.unscale_V(phi_n))
    phi_p_unscaled = np.array(scaling.unscale_V(phi_p))
    
    # Energy bands (eV). Reference is intrinsic level E_i = -q * psi
    # E_c = E_i + E_g/2, E_v = E_i - E_g/2. Approximate E_g ~ 1.12 eV for Si
    E_g = 1.12
    E_i = -psi
    E_c = E_i + E_g / 2
    E_v = E_i - E_g / 2
    E_Fn = -phi_n_unscaled
    E_Fp = -phi_p_unscaled
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
    
    # Plot 1: Energy Bands
    ax1.plot(x_um, E_c, 'k-', label='$E_c$')
    ax1.plot(x_um, E_v, 'k-', label='$E_v$')
    ax1.plot(x_um, E_i, 'k--', alpha=0.5, label='$E_i$')
    ax1.plot(x_um, E_Fn, 'b--', label='$E_{Fn}$')
    ax1.plot(x_um, E_Fp, 'r--', label='$E_{Fp}$')
    
    ax1.set_ylabel('Energy [eV]')
    ax1.set_title(f'{title} - Energy Bands')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Carrier Concentrations
    ax2.semilogy(x_um, n, 'b-', label='n')
    ax2.semilogy(x_um, p, 'r-', label='p')
    
    # Plot doping profiles (absolute values)
    N_D_plot = np.where(N_dop > 0, N_dop, np.nan)
    N_A_plot = np.where(N_dop < 0, -N_dop, np.nan)
    
    ax2.semilogy(x_um, N_D_plot, 'b--', alpha=0.3, label='$N_D$')
    ax2.semilogy(x_um, N_A_plot, 'r--', alpha=0.3, label='$N_A$')
    
    ax2.set_ylabel('Concentration [cm$^{-3}$]')
    # Convert m^-3 to cm^-3 for standard semiconductor plots
    ax2.set_yticklabels(['10$^{%d}$' % int(np.log10(y)-6) for y in ax2.get_yticks()])
    ax2.set_xlabel('Position [$\mu$m]')
    ax2.set_title('Carrier & Doping Concentrations')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {filename}")
    else:
        plt.show()
    
    plt.close()

def plot_iv_curve(V_array, J_array, title="I-V Characteristic", filename=None):
    """
    Plots the Current-Voltage (I-V) characteristic curve.
    """
    plt.figure(figsize=(6, 4))
    
    # Standard convention: Forward bias is positive current and positive voltage
    # Depending on how the junction is defined, we might need to flip the sign
    # In our case, N_A is on the right, so positive voltage on the right is forward bias,
    # but the current flows left (negative). So we plot -J vs V for standard orientation.
    J_plot = -np.array(J_array)
    V_plot = np.array(V_array)
    
    plt.semilogy(V_plot[J_plot > 0], J_plot[J_plot > 0], 'b.-', label='Forward Bias')
    plt.semilogy(V_plot[J_plot < 0], -J_plot[J_plot < 0], 'r.-', label='Reverse Bias (abs)')
    
    plt.xlabel('Applied Voltage [V]')
    plt.ylabel('|Current Density| [A/cm$^2$]')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {filename}")
    else:
        plt.show()
    plt.close()
