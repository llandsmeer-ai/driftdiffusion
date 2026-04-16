import jax.numpy as jnp
import chex

@chex.dataclass
class Grid:
    x: jnp.ndarray      # Node coordinates (dimensionless)
    dx: jnp.ndarray     # Node spacings (x[i+1] - x[i])
    x_mid: jnp.ndarray  # Midpoint coordinates
    dx_mid: jnp.ndarray # Dual grid spacings (distance between midpoints or boundary to midpoint)

def generate_nonuniform_mesh(L, n_points, junction_pos, refinement_factor=3.0):
    """
    Generates a 1D non-uniform mesh clustered around a junction position.
    Args:
        L: Total length of the device
        n_points: Number of nodes
        junction_pos: Position of the junction
        refinement_factor: Controls how dense the mesh is at the junction
    Returns:
        Grid object containing the mesh points and spacings.
    """
    # Create two geometric progressions meeting at the junction
    n_left = int(n_points * (junction_pos / L))
    n_right = n_points - n_left
    
    # Left side: cluster near junction_pos
    # t goes from 0 to 1
    t_left = jnp.linspace(0, 1, n_left)
    # Power law clustering: higher power means more clustering near t=1
    x_left = junction_pos * (1.0 - (1.0 - t_left)**refinement_factor)
    
    # Right side: cluster near junction_pos
    t_right = jnp.linspace(0, 1, n_right)[1:] # Exclude 0 to avoid duplicate junction point
    x_right = junction_pos + (L - junction_pos) * (t_right**refinement_factor)
    
    x = jnp.concatenate([x_left, x_right])
    
    dx = jnp.diff(x)
    x_mid = x[:-1] + dx / 2.0
    
    # Dual cell width for finite volume method (dx_mid)
    # For interior nodes i: dx_mid[i] = (x_{i+1} - x_{i-1}) / 2 = dx[i-1]/2 + dx[i]/2
    # For boundary nodes: dx_mid[0] = dx[0]/2, dx_mid[-1] = dx[-1]/2
    dx_mid = jnp.zeros_like(x)
    dx_mid = dx_mid.at[1:-1].set((dx[:-1] + dx[1:]) / 2.0)
    dx_mid = dx_mid.at[0].set(dx[0] / 2.0)
    dx_mid = dx_mid.at[-1].set(dx[-1] / 2.0)
    
    return Grid(x=x, dx=dx, x_mid=x_mid, dx_mid=dx_mid)
