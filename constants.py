import jax.numpy as jnp

# Physical Constants (MKS units)
q = 1.60217663e-19       # Elementary charge [C]
eps_0 = 8.8541878128e-12 # Vacuum permittivity [F/m]
k_B = 1.380649e-23       # Boltzmann constant [J/K]

class DeMariScaling:
    def __init__(self, T=300.0, ni=1.0e16, eps_r=11.7, mu_n_si=0.1):
        """
        De Mari scaling factors to non-dimensionalize drift-diffusion equations.
        Args:
            T: Temperature [K]
            ni: Intrinsic carrier concentration [m^-3] (Si ~1e16 m^-3 at 300K)
            eps_r: Relative permittivity (Si ~11.7)
            mu_n_si: Electron mobility [m^2/Vs] (used for characteristic time/current)
        """
        self.T = T
        self.ni = ni
        self.eps = eps_r * eps_0
        self.mu_n_si = mu_n_si
        
        self.V_T = k_B * T / q
        self.L_D = jnp.sqrt(self.eps * self.V_T / (q * self.ni))
        
        self.D_n_si = self.mu_n_si * self.V_T
        self.tau_c = self.L_D**2 / self.D_n_si
        
        # Scaling factors (multiply by these to get SI units)
        self.x_scale = self.L_D
        self.V_scale = self.V_T
        self.C_scale = self.ni
        self.t_scale = self.tau_c
        self.J_scale = (q * self.ni * self.D_n_si) / self.L_D
        
    def scale_x(self, x_si): return x_si / self.x_scale
    def unscale_x(self, x_dl): return x_dl * self.x_scale
    
    def scale_V(self, V_si): return V_si / self.V_scale
    def unscale_V(self, V_dl): return V_dl * self.V_scale
    
    def scale_C(self, C_si): return C_si / self.C_scale
    def unscale_C(self, C_dl): return C_dl * self.C_scale
    
    def unscale_J(self, J_dl): return J_dl * self.J_scale
