import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

def simulate_2d_sde(sde, x0, t_start, t_end, dt, key):
    D = 2
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



def run_sde_verification(sde_bridge, t_array, num_samples=100, filename ="sde_gp_verification.png"):
    # 1. Generate paths: This now returns an array of shape (num_samples, num_steps, D)
    key = jax.random.PRNGKey(42)
    # If your sample_trajectories returns an array directly:
    paths = sde_bridge.sample_trajectories(num_samples, t_array, key)
    
    # 2. Extract statistics
    mu_vals = jax.vmap(sde_bridge.mean_fn)(t_array)
    cov_vals = jax.vmap(sde_bridge.Sigma_fn)(t_array)
    std_vals = jnp.sqrt(jax.vmap(jnp.diag)(cov_vals))

    # 3. Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    dim_names = ['x1', 'x2']
    
    for i in range(2):
        ax = axes[i]
        # True Mean
        ax.plot(t_array, mu_vals[:, i], 'k--', label='True Mean', lw=2)
        
        # 95% Confidence Interval (2*std)
        ax.fill_between(t_array, 
                        mu_vals[:, i] - 2*std_vals[:, i], 
                        mu_vals[:, i] + 2*std_vals[:, i], 
                        color='gray', alpha=0.3, label='95% Confidence')
        
        # Plot simulated paths (using vmap-friendly indexing)
        for n in range(num_samples):
            ax.plot(t_array, paths[n, :, i], color='red', alpha=1, lw=0.5)
            
        ax.set_title(f'Dimension {dim_names[i]}')
        ax.grid(True)
        
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Verification plot with {num_samples} paths saved to {filename}.")

def run_sde_verification_old(sde_bridge, t_array, num_samples=100):
    dt = t_array[1] - t_array[0]
    
    # Prepare parallel simulation
    # vmap over the 'key' argument to simulate different paths in parallel
    keys = jax.random.split(jax.random.PRNGKey(42), num_samples)
    
    # Define a partial function to get initial states for N samples
    def get_sample_path(key):
        k1, k2 = jax.random.split(key)
        x0 = sde_bridge.get_initial_state(k1)
        return simulate_2d_sde(sde_bridge, x0, t_array[0], t_array[-1], dt, k2)

    # Run all simulations in parallel
    paths = jax.vmap(get_sample_path)(keys) # Result shape: (num_samples, num_steps, D)
    
    # Extract statistics
    mu_vals = jax.vmap(sde_bridge.mean_fn)(t_array)
    cov_vals = jax.vmap(sde_bridge.Sigma_fn)(t_array)
    std_vals = jnp.sqrt(jax.vmap(jnp.diag)(cov_vals))
    
    # Setup Plot
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
        
        # Plot all  paths
        for n in range(num_samples):
            ax.plot(t_plot_sim, np.array(paths[n, :, i]), color='red', alpha=1, lw=0.5)
            
        ax.set_title(f'Dimension {dim_names[i]}')
        ax.grid(True)
        
    plt.tight_layout()
    plt.savefig("sde_simulation_comparison.png")
    print(f"Verification plot with {num_samples} paths saved.")
