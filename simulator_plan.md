# 1D Drift-Diffusion Semiconductor Simulator Plan

## Overview
A robust, expert-level 1D drift-diffusion solver in Python using JAX for JIT compilation, automatic differentiation (for exact Jacobians), and GPU/CPU acceleration. The simulator handles steady-state and transient simulations of P-N and PIN junctions, with an architecture designed to be extensible to slow-moving mobile ions in the future.

## 1. Physics & Scaling (De Mari Scaling)
Semiconductor equations are notoriously stiff due to exponential dependencies and large variations in carrier concentrations and doping. We must use De Mari scaling to non-dimensionalize the equations:
*   **Lengths ($x$):** Scaled by intrinsic Debye length $L_D = \sqrt{\epsilon k_B T / (q^2 n_i)}$
*   **Potentials ($\psi$):** Scaled by thermal voltage $V_T = k_B T / q$
*   **Concentrations ($n, p, N_D, N_A$):** Scaled by intrinsic carrier concentration $n_i$
*   **Time ($t$):** Scaled by characteristic time $\tau = L_D^2 / D_n$ (or similar)

**Dimensionless Equations:**
1.  **Poisson:** $\nabla^2 \psi = n - p - N_{net}$  (where $N_{net} = N_D - N_A$)
2.  **Electron Continuity:** $\frac{\partial n}{\partial t} = \nabla \cdot J_n - R$
3.  **Hole Continuity:** $\frac{\partial p}{\partial t} = -\nabla \cdot J_p - R$

**Dimensionless Current Densities:**
*   $J_n = \mu_n (n \nabla \psi + \nabla n)$
*   $J_p = \mu_p (p \nabla \psi - \nabla p)$
*(Note: scaled mobilities/diffusivities will be used, often $D = \mu$ due to scaled Einstein relation).*

## 2. Numerical Discretization
*   **Grid:** 1D Non-uniform staggered grid (Finite Volume Method). Nodes hold scalars ($\psi, n, p$), edges hold fluxes ($J_n, J_p, E$). Grid must be dense near junctions.
*   **Scharfetter-Gummel (SG) Scheme:** Mandatory for stability.
    *   $J_{n, i+1/2} = \frac{\mu_n}{\Delta x} \left[ n_{i+1} B(\Delta \psi) - n_i B(-\Delta \psi) \right]$
    *   $J_{p, i+1/2} = \frac{\mu_p}{\Delta x} \left[ p_{i} B(\Delta \psi) - p_{i+1} B(-\Delta \psi) \right]$
    *   Where $\Delta \psi = \psi_{i+1} - \psi_i$ and $B(x) = x / (e^x - 1)$ is the Bernoulli function.
*   **Safe Bernoulli:** A Taylor-expanded version of $B(x)$ near $x=0$ is required to avoid `NaN` gradients during JAX automatic differentiation.
*   **Time Stepping:** Backward Euler (Implicit) for unconditional A-stability in transients.

## 3. JAX Architecture & Solver
*   **Data Structures:** Pure functional design using immutable structures (e.g., `typing.NamedTuple` or `chex.dataclass`) for `Grid`, `Material`, and `State` (holding $\psi, n, p$).
*   **Precision:** MUST use `jax.config.update("jax_enable_x64", True)`.
*   **Residual Function:** A pure function `compute_residuals(state, old_state, dt, ...)` that calculates the mismatch in Poisson and Continuity equations.
*   **Automatic Differentiation:** Use `jax.jacfwd` to automatically compute the exact Jacobian of the residual function. This eliminates manual Jacobian derivation errors.
*   **Newton-Raphson Solver:** 
    *   Fully coupled (solves $\psi, n, p$ simultaneously).
    *   JIT-compiled `jax.lax.while_loop`.
    *   Block-tridiagonal linear solver using `jax.scipy.linalg.solve_banded` for $O(N)$ scaling.
    *   Damping applied to potential updates ($\Delta \psi$) to prevent carrier concentrations from becoming negative during aggressive steps.

## 4. Execution Plan
*   **Step 1:** Setup the constants, De Mari scaling functions, and Grid generation (non-uniform meshing).
*   **Step 2:** Implement the safe Bernoulli function, SRH recombination, and the `compute_residuals` physics function.
*   **Step 3:** Implement the JIT-compiled Newton-Raphson solver using `jax.jacfwd`.
*   **Step 4:** Write boundary condition handlers (Ohmic contacts, Dirichlet).
*   **Step 5:** Create the Steady-State initialization routine (thermal equilibrium -> applied bias).
*   **Step 6:** Create the Transient loop (`jax.lax.scan`) for time evolution.
*   **Step 7:** Run a test case: Silicon P-N junction with a voltage step to observe capacitive and diffusion currents.

## 5. Future Extensibility (Mobile Ions)
*   Ions operate on much slower timescales.
*   Can be added using Operator Splitting: 
    1. Freeze ion profile.
    2. Solve fast electronic steady-state (or quasi-steady-state).
    3. Calculate ion fluxes using new potential.
    4. Step ions forward in time.
    5. Repeat.