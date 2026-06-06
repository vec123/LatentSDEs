import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from src.sde.GaussianPaths.gaussian_path import GaussianPathSDE
from src.sde.tests.gp_ground_truths import get_ground_truth_cov, get_ground_truth_mean
from src.sde.tests.plot_helper import plot_covar_components_dxd


# Diffusion matrix (2x2)
def G_fn(t):
    return jnp.array([[0.8, 0.0], [0.1, 0.8]])

def simulate_2d_sde(sde, x0, t_start, t_end, dt, key):
    steps = int((t_end - t_start) / dt)
    t = t_start
    x = x0
    
    # Storage for analysis
    path = []
    for i in range(steps):
        key, subkey = jax.random.split(key)
        # SDE step: dx = (Ax + b)dt + G dW
        drift = sde.drift(t, x)
        diffusion = sde.diffusion(t)
        
        noise = jax.random.normal(subkey, (D,))
        x = x + drift * dt + (diffusion @ noise) * jnp.sqrt(dt)
        
        path.append(x)
        t += dt
    return jnp.array(path)



def run_sde_verification(sde_bridge, t_array, num_samples=100):
    dt = t_array[1] - t_array[0]
    
    # 1. Prepare parallel simulation
    # vmap over the 'key' argument to simulate different paths in parallel
    keys = jax.random.split(jax.random.PRNGKey(42), num_samples)
    
    # Define a partial function to get initial states for N samples
    def get_sample_path(key):
        k1, k2 = jax.random.split(key)
        x0 = sde_bridge.get_initial_state(k1)
        return simulate_2d_sde(sde_bridge, x0, t_array[0], t_array[-1], dt, k2)

    # Run all simulations in parallel
    paths = jax.vmap(get_sample_path)(keys) # Result shape: (num_samples, num_steps, D)
    
    # 2. Extract statistics
    mu_vals = jax.vmap(sde_bridge.mean_fn)(t_array)
    cov_vals = jax.vmap(sde_bridge.Sigma_fn)(t_array)
    std_vals = jnp.sqrt(jax.vmap(jnp.diag)(cov_vals))
    
    # 3. Setup Plot
    num_steps = paths.shape[1]
    t_plot_sim = np.linspace(t_array[0], t_array[-1], num_steps)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    dim_names = ['x', 'y']
    
    for i in range(2):
        ax = axes[i]
        ax.plot(t_plot_sim, np.array(mu_vals[:num_steps, i]), 'k--', label='True Mean', lw=2)
        ax.fill_between(t_plot_sim, 
                        np.array(mu_vals[:num_steps, i]) - 2*np.array(std_vals[:num_steps, i]), 
                        np.array(mu_vals[:num_steps, i]) + 2*np.array(std_vals[:num_steps, i]), 
                        color='gray', alpha=0.3, label='95% Confidence')
        
        # Plot all 100 paths with high transparency
        for n in range(num_samples):
            ax.plot(t_plot_sim, np.array(paths[n, :, i]), color='red', alpha=1, lw=0.5)
            
        ax.set_title(f'Dimension {dim_names[i]}')
        ax.grid(True)
        
    plt.tight_layout()
    plt.savefig("sde_simulation_comparison.png")
    print(f"Verification plot with {num_samples} paths saved.")

if __name__ == "__main__":
    D = 2
    T = 10.0 
    eps = 1e-6
    NUM_TIMES = 500
    t_array = jnp.linspace(0, T, NUM_TIMES)

    mu_true = jax.vmap(get_ground_truth_mean)(t_array)
    S_true = jax.vmap(get_ground_truth_cov)(t_array)
    sde_bridge = GaussianPathSDE(
        mean_fn=get_ground_truth_mean,
        # Note: You need a wrapper to make the Jacobian of the function work
        dmean_dt_fn=lambda t: jax.jacobian(get_ground_truth_mean)(t),
        Sigma_fn=get_ground_truth_cov,
        dSigma_dt_fn=lambda t: jax.jacobian(get_ground_truth_cov)(t),
        G_fn=G_fn
    )
    run_sde_verification(sde_bridge,t_array,100)