import jax
import jax.numpy as jnp

import jax
import jax.numpy as jnp
from diffrax import  diffeqsolve, ODETerm, Tsit5, SaveAt

def generate_data(num_trajectories=5, num_points=40):
    t = jnp.linspace(0, 4 * jnp.pi, num_points)
    trajs = []
    for i in range(num_trajectories):
        # Sine wave with random phase and noise
        y = jnp.sin(t) + jax.random.normal(jax.random.PRNGKey(i), t.shape) * 0.4
        trajs.append((t, y))
    return trajs


def van_der_pol_dynamics(t, y, args):
    mu = 1.5  # Nonlinear damping parameter
    x1, x2 = y
    dx1 = x2
    dx2 = mu * (1 - x1**2) * x2 - x1
    return jnp.array([dx1, dx2])

def generate_vanderpol_data(num_trajectories=5, num_points=60, mu=1.5):
    T = 20  # Total time
    dt = 0.1
    
    t_eval = jnp.linspace(0, T, num_points)
    trajs = []
    
    # Define different initial conditions for x1
    init_x1 = jnp.linspace(-0.1, 0.1, num_trajectories)
    
    for i in range(num_trajectories):
        y0 = jnp.array([init_x1[i], 0.0]) # Start with x1=init, x2=0
        term = ODETerm(van_der_pol_dynamics)
        solver = Tsit5()
        
        # Integrate the ODE
        sol = diffeqsolve(term, solver, t0=0, t1=T, dt0=dt, y0=y0, 
                          saveat=SaveAt(ts=t_eval))
        
        # Extract only x1 (the first state variable)
        x1_traj = sol.ys[:, 0]
        # Add slight observation noise
        noisy_x1 = x1_traj + jax.random.normal(jax.random.PRNGKey(i), x1_traj.shape) * 0.05
        
        trajs.append((t_eval, noisy_x1))
        
    return trajs