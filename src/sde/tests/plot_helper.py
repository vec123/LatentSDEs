import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from src.sde.tests.obstacles import get_obstacle_loss
import jax
import numpy as np
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def save_fit_plot(model, t_points, y_target, filename):
    t_dense = jnp.linspace(t_points.min(), t_points.max(), 100)
    y_pred = jax.vmap(model)(t_dense)
    dy_pred = jax.vmap(model.get_dot_mu)(t_dense)
    print(f"dy_pred shape:", dy_pred.shape)  # Debugging line to check shape of dy_pred
    print(f"dy_pred values:", dy_pred)  # Debugging line to check values of dy_pred
    #dy_pred = jax.vmap(jax.jacobian(model))(t_dense)   
    plt.figure(figsize=(8, 4))
    plt.scatter(t_points, y_target, color='red', label='Data', alpha=0.5)
    plt.plot(t_dense, y_pred, label='Mean Prediction', linewidth=2)
    plt.plot(t_dense, dy_pred, label='Derivative (dot_mu)', linestyle=':', linewidth=2)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
    print(f"Plot saved to {filename}")

def plot_ensemble(model, trajs, title, filename, eps = 1e-6):
    plt.figure(figsize=(10, 6))
    
    # Plot all trajectories
    for i, (t, y) in enumerate(trajs):
        if i <10:
            plt.plot(t, y, 'gray', alpha=0.3, label='Ground Truth' if t is trajs[0][0] else "")
    
    # Plot mean
    t_dense = jnp.linspace(trajs[0][0].min()+eps, trajs[0][0].max()-eps, 100)
    # Using list comprehension to bypass vmap/tracer issues for generic model types
    y_pred = jnp.array([model(ti) for ti in t_dense])
    
    plt.plot(t_dense, y_pred, 'r-', linewidth=2, label='Model Mean Prediction')
    plt.title(title)
    plt.legend()
    plt.savefig(filename)
    plt.close()



def plot_and_save_potential(obstacles, file_name="potential_field.png"):
    # 1. Create a grid
    x = np.linspace(-1.5, 1.5, 200)
    y = np.linspace(-1.5, 1.5, 200)
    X, Y = np.meshgrid(x, y)
    
    # 2. Prepare grid points (200, 200, 2)
    grid_points = np.stack([X, Y], axis=-1)
    
    # 3. Compute potential
    # Use vmap to compute the loss across the entire grid
    # We use jnp to avoid JAX/NumPy type conflicts inside the vmap
    @jax.vmap
    @jax.vmap
    def compute_pot(point):
        return get_obstacle_loss(point, obstacles)
    
    Z = compute_pot(grid_points)
    
    # FIX: Ensure Z is 2D (200, 200) by squeezing out any extra dimensions
    Z = np.array(Z).squeeze() 
    
    # 4. Plotting
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, Z, levels=50, cmap='inferno')
    plt.colorbar(label='Obstacle Loss Value')
    
    for obs in obstacles:
        circle = plt.Circle(obs['center'], obs['radius'], color='white', fill=False, linewidth=2)
        plt.gca().add_patch(circle)
        
    plt.title("Optimizer Obstacle Landscape")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(file_name)
    print(f"Plot saved to {file_name}")

def plot_final_path(t_points, mean_model, obstacles, start_pos, end_pos, file_name="final_path.png"):
            # final_cov = MaternCovariance(D=2, time_interval=(0.0, T), num_basis=NUM_SUPPORT)
            path = jax.vmap(mean_model)(t_points)
            #cov = jax.vmap(lambda t: final_cov.get_cov(t, params_R, params_L, 1.0, 1.0, 1.0))(t_points)

            plt.figure(figsize=(8, 8))
            for obs in obstacles:
                plt.gca().add_patch(plt.Circle(obs['center'], obs['radius'], color='r', alpha=0.3))
                
            plt.plot(path[:, 0], path[:, 1], 'b-', label='Optimized Mean Path')
            plt.scatter(path[:, 0], path[:, 1], c='blue', s=10, label='Path Points')
            plt.scatter([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], color='green', label='Boundary')
            plt.legend(); plt.grid(True); plt.savefig(file_name)

def plot_model_path(t_eval, mean_model, y_truth = None, file_name="model_path.png"):
    
    path = jax.vmap(mean_model)(t_eval)
    dpath = jax.vmap(mean_model.get_dot_mu)(t_eval)

    # Create a 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Spatial Trajectory (X vs Y)
    axes[0, 0].plot(path[:, 0], path[:, 1], 'b-')
    axes[0, 0].scatter(path[:, 0], path[:, 1], color='green', s=10, zorder=3)
    axes[0, 0].quiver(path[::5, 0], path[::5, 1], dpath[::5, 0], dpath[::5, 1], color='r', alpha=0.5)
    axes[0, 0].set_title("Spatial Trajectory (X-Y)")
    axes[0, 0].grid(True)

    # 2. X position over time
    axes[0, 1].plot(t_eval, path[:, 0], 'b-', label='X(t)')
    if y_truth is not None:
         axes[0, 1].plot(t_eval, y_truth[:, 0], 'r--', alpha = 0.5, label='X_true(t)')
    axes[0, 1].set_title("X position over time")
    axes[0, 1].grid(True)

    # 3. Y position over time
    axes[1, 0].plot(t_eval, path[:, 1], 'g-', label='Y(t)')
    if y_truth is not None:
         axes[1, 0].plot(t_eval, y_truth[:, 1], 'r--',  alpha = 0.5, label='Y_true(t)')
    axes[1, 0].set_title("Y position over time")
    axes[1, 0].grid(True)

    # 4. Velocities over time
    axes[1, 1].plot(t_eval, dpath[:, 0], 'r--', label='dX/dt')
    axes[1, 1].plot(t_eval, dpath[:, 1], 'm--', label='dY/dt')
    axes[1, 1].set_title("Velocity over time")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()
    print(f"Plot saved to {file_name}")


def plot_knots_and_coeffs(mean_model, file_name="knots_and_coeffs.png"):
    """
    Plots the knot vector as vertical lines and the coefficients 
    at their respective temporal influence centers.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    knots = mean_model.knots_padded
    coeffs = mean_model.coeffs
    k = 3  # The degree of your spline
    
    # Calculate influence centers (Schoenberg-Whitney points)
    # The i-th coefficient is associated with the average of k knots
    knot_centers = [jnp.mean(knots[i+1 : i+k+1]) for i in range(len(coeffs))]
    
    titles = ["X-Coefficients", "Y-Coefficients"]
    for dim in range(2):
        # 1. Plot knots as vertical lines
        for knot in knots:
            axes[dim].axvline(x=knot, color='gray', linestyle=':', alpha=0.5, label='Knots' if knot == knots[0] else "")
        
        # 2. Plot coefficients
        axes[dim].plot(knot_centers, coeffs[:, dim], 'ro-', label='Coeffs')
        axes[dim].set_ylabel(f"Value in Dim {dim}")
        axes[dim].set_title(titles[dim])
        axes[dim].grid(True, alpha=0.3)
        axes[dim].legend()

    plt.xlabel("Time (t)")
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()
    print(f"Structure plot saved to {file_name}")


def plot_static_path_visualization(mean_model, obstacles, T, file_name="static_path_viz.png"):
    """
    Plots the trajectory with time-markers to visualize velocity/spacing.
    """
    plt.figure(figsize=(8, 8))
    
    # 1. Setup scene
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    for obs in obstacles:
        plt.gca().add_patch(plt.Circle(obs['center'], obs['radius'], color='r', alpha=0.3))
    
    # 2. Plot the full path
    t_dense = jnp.linspace(0, T, 200)
    full_path = jax.vmap(mean_model)(t_dense)
    plt.plot(full_path[:, 0], full_path[:, 1], 'b-', alpha=0.5, label='Path')
    
    # 3. Plot markers at regular time intervals to show velocity
    # Closer markers = slower speed, Further markers = higher speed
    num_markers = 20
    t_markers = jnp.linspace(0, T, num_markers)
    marker_positions = jax.vmap(mean_model)(t_markers)
    
    plt.scatter(marker_positions[:, 0], marker_positions[:, 1], 
                c=t_markers, cmap='viridis', s=50, zorder=5, label='Time Markers')
    plt.colorbar(label='Time (t)')
    
    plt.legend()
    plt.grid(True)
    plt.title("Static Path Visualization (Color = Time)")
    plt.savefig(file_name)
    plt.close()
    print(f"Static plot saved to {file_name}")

def plot_trajs(trajs, filename="trajs.png"):
    
    _, first_y = trajs[0]
    D = first_y.shape[1] if first_y.ndim > 1 else 1
    
    # Setup subplots based on D
    fig, axes = plt.subplots(D, 1, figsize=(10, 3 * D), squeeze=False)
    
    for t_i, y_i in trajs:
        for d in range(D):
            # Extract dimension d, handling both 1D and 2D arrays
            y_d = y_i[:, d] if y_i.ndim > 1 else y_i
            axes[d, 0].scatter(t_i, y_d, color='gray', alpha=0.7, s=1)
            axes[d,0].plot(t_i, y_d, color='gray', alpha=0.3)
            axes[d, 0].set_title(f"Dimension {d+1}")
            axes[d, 0].grid(True)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Static plot saved to {filename}")
def plot_gp(t_array, mean_fn, cov_fn, filename="gp_definition.png"):
    # 1. Compute stats across the time array
    # We vmap to get mean (NUM_TIMES, 2) and covariance (NUM_TIMES, 2, 2)
    mu_vals = jax.vmap(mean_fn)(t_array)
    cov_vals = jax.vmap(cov_fn)(t_array)
    
    # Extract standard deviations from the diagonal
    std_vals = jnp.sqrt(jax.vmap(jnp.diag)(cov_vals))
    
    # 2. Setup plotting
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    labels = ["x1 (Position)", "x2 (Velocity)"]
    
    for i in range(2):
        # Mean
        axes[i].plot(t_array, mu_vals[:, i], 'r-', label="GP Mean", lw=2)
        
        # 95% Confidence Interval (2*std)
        axes[i].fill_between(t_array, 
                             mu_vals[:, i] - 2*std_vals[:, i], 
                             mu_vals[:, i] + 2*std_vals[:, i], 
                             color='red', alpha=0.2, label="95% Confidence")
        
        axes[i].set_ylabel(labels[i])
        axes[i].legend()
        axes[i].grid(True)
        
    axes[1].set_xlabel("Time (t)")
    plt.suptitle("Gaussian Process Definition (Mean & Variance)")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"GP definition plot saved to {filename}")


def plot_gp_ground_truth(trajs, t_vals, dim=2):
    """
    Plots the ground truth mean and covariance evolution against sampled trajectories.
    """
    # 1. Calculate ground truth statistics over the entire time range
    mu_true, S_true = jax.vmap(get_ground_truth)(t_vals)
    
    # 2. Setup figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # --- Plot A: Trajectories and Mean ---
    # Plot trajectories
    for t_i, y_i in trajs:
        axes[0].scatter(t_i, y_i[:, 0], color='gray', alpha=0.1)
    
    # Plot True Mean
    axes[0].plot(t_vals, mu_true[:, 0], 'r--', linewidth=2, label='True Mean (X)')
    axes[0].set_title("Ground Truth: Trajectories & Mean (X)")
    axes[0].legend()
    
    # --- Plot B: Covariance Evolution (Diagonal) ---
    axes[1].plot(t_vals, S_true[:, 0, 0], 'b-', label='Var(X)')
    axes[1].plot(t_vals, S_true[:, 1, 1], 'g-', label='Var(Y)')
    axes[1].set_title("Ground Truth: Variance (Diagonal)")
    axes[1].legend()
    
    # --- Plot C: Covariance Evolution (Off-Diagonal) ---
    axes[2].plot(t_vals, S_true[:, 0, 1], 'purple', label='Cov(XY)')
    axes[2].set_title("Ground Truth: Covariance (Off-Diag)")
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig("ground_truth.png")
    print("Ground truth plot saved to ground_truth.png")


def plot_gp_comparison(t_vals, mu_true, S_true, mu_pred, S_pred,  dim=2, trajs=None,filename = "matern_comparison.png"):
    data_dim =dim
    # Evaluate Models

    # 3. Plot
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    # Plot Means + Sampled Trajectories
    if trajs is not None:
        for t_i, y_i in trajs:
            #axes[0].scatter(t_i, y_i[:, 0], color='black', alpha=1, s=5)
            axes[0].plot(t_i, y_i[:, 0], color='black', alpha=0.5)
            
    axes[0].plot(t_vals, mu_true[:, 0], 'k--', label='True X Mean')
    axes[0].plot(t_vals, mu_pred[:, 0], 'b-', label='Matern X Mean')
    std_x_model = jnp.sqrt(S_pred[:, 0, 0])
    axes[0].fill_between(t_vals, mu_pred[:, 0] - 2*std_x_model, mu_pred[:, 0] + 2*std_x_model, color='red', alpha=0.2, label='Model 2σ')
    std_x_true = jnp.sqrt(S_true[:, 0, 0])
    axes[0].fill_between(t_vals, mu_true[:, 0] - 2*std_x_true, mu_true[:, 0] + 2*std_x_true, color='blue', alpha=0.2, label='True 2σ')  
    axes[0].set_title("Mean Evolution")
    axes[0].legend()
    
    if data_dim > 1:
        if trajs is not None:
            for t_i, y_i in trajs:
                #axes[1].plot(t_i, y_i[:, 1], color='black', alpha=0.5, s=5)
                axes[1].plot(t_i, y_i[:, 1], color='black', alpha=0.5)
        axes[1].plot(t_vals, mu_true[:, 1], 'k--', label='True Y Mean')
        axes[1].plot(t_vals, mu_pred[:, 1], 'b-', label='Matern Y Mean')
        std_y_model = jnp.sqrt(S_pred[:, 1, 1])
        axes[1].fill_between(t_vals, mu_pred[:, 1] - 2*std_y_model, mu_pred[:, 1] + 2*std_y_model, color='red', alpha=0.2, label='Model 2σ')
        std_y_true = jnp.sqrt(S_true[:, 1, 1])
        axes[1].fill_between(t_vals, mu_true[:, 1] - 2*std_y_true, mu_true[:, 1] + 2*std_y_true, color='blue', alpha=0.2, label='True 2σ')
        axes[1].set_title("Mean Evolution")
        axes[1].legend()
    

    # Plot Covariance Diagonal + Confidence Interval
    axes[2].plot(t_vals, S_true[:, 0, 0], 'b-', label='True Var(X)')
    axes[2].plot(t_vals, S_pred[:, 0, 0], 'r-', label='Matern Var(X)')
    axes[2].set_title("Variance (X)")
    axes[2].legend()
    
    if data_dim > 1:
        # Plot Off-Diagonal
        axes[3].plot(t_vals, S_true[:, 0, 1], 'b-', label='True Cov(XY)')
        axes[3].plot(t_vals, S_pred[:, 0, 1], 'r-', label='Matern Cov(XY)')
        axes[3].set_title("Off-Diagonal Covariance")
        axes[3].legend()
    
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Comparison plot saved with samples to {filename}")

def plot_cov_true_empirical_learned(trajs, t_vals, mu_true, S_true, mean_model, cov_model, W_R, W_L):
    """
    trajs: List of (t_i, y_i) where y_i shape is (num_samples, dim)
    mean_fn: Function of t returning true mean (dim,)
    cov_fn: Function of t returning true covariance (dim, dim)
    """
    
    #  Compute empirical stats at each t
    def compute_empirical_moments(trajs):
        empirical_means = []
        empirical_covs = []
        
        for t in t_vals:
            # Collect all samples at time t across all trajectories
            # Assuming trajs is a list of (t_array, y_array)
            samples_at_t = []
            for t_arr, y_arr in trajs:
                # Find index where t_arr == t (or closest)
                idx = jnp.argmin(jnp.abs(t_arr - t))
                samples_at_t.append(y_arr[idx])
            
            samples_at_t = jnp.stack(samples_at_t) # (num_trajs, dim)
            
            # Empirical mean and covariance
            e_mean = jnp.mean(samples_at_t, axis=0)
            e_cov = jnp.cov(samples_at_t, rowvar=False)
            
            empirical_means.append(e_mean)
            empirical_covs.append(e_cov)
        return jnp.stack(empirical_means), jnp.stack(empirical_covs)
        
    empirical_means, empirical_covs = compute_empirical_moments(trajs)

    # Get true stats
    # true_means, true_covs = jax.vmap(mu_true)(t_vals), jax.vmap(S_true)(t_vals)
    mu_pred = jax.vmap(mean_model)(t_vals)
    S_pred = jax.vmap(lambda t: cov_model.get_cov(t, W_R, W_L, 1.0, 0.2, 1.0))(t_vals)
    
    # 3. Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Mean Plot (X-dimension)
    axes[0].plot(t_vals, mu_true[:, 0], 'b--', label='True Mean')
    axes[0].plot(t_vals, mu_pred[:, 0], 'k--', label='Model Mean')
    axes[0].plot(t_vals, empirical_means[:, 0], 'ro', alpha=0.5, label='Empirical Mean')
    axes[0].set_title("Mean Comparison (X)")
    axes[0].legend()
    
    # Covariance Plot (Var X)
    axes[1].plot(t_vals, S_true[:, 0, 0], 'b--', label='True Var(X)')
    axes[1].plot(t_vals, S_pred[:, 0, 0], 'k--', label='Model Var(X)')
    axes[1].plot(t_vals, empirical_covs[:, 0, 0], 'ro', alpha=0.5, label='Empirical Var(X)')
    axes[1].set_title("Variance Comparison (X)")
    axes[1].legend()
    
    plt.savefig("empirical_vs_true.png")

def plot_covar_components_2x2(t_values, cov_matrices, filename = "covar_components.png"):
    """
    t_values: array of time points
    cov_matrices: array of shape (len(t_values), 2, 2)
    """
    # Components to plot: (row, col)
    # For a 2x2 symmetric matrix, unique DOFs are (0,0), (0,1), (1,1)
    dofs = [(0, 0), (0, 1), (1, 1)]
    labels = ["Var_X", "Cov_XY", "Var_Y"]
    
    for (r, c), label in zip(dofs, labels):
        plt.figure(figsize=(8, 4))
        # Extract the specific component across all time steps
        values = cov_matrices[:, r, c]
        
        plt.plot(t_values, values, label=label, color='tab:blue', linewidth=2)
        plt.title(f"Evolution of {label}")
        plt.xlabel("Time (t)")
        plt.ylabel("Value")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Save each figure
        plt.savefig(filename)
        plt.close()
        print(f"Saved: {filename}")

def plot_covar_components_dxd(t_values, cov_matrices, filename ="covar_components.png"):
    dim = cov_matrices.shape[1]
    rows, cols = jnp.triu_indices(dim)
    num_plots = len(rows)
    
    # Create one figure with a subplot for each degree of freedom
    fig, axes = plt.subplots(num_plots, 1, figsize=(8, 3 * num_plots), constrained_layout=True)
    
    # Ensure axes is iterable even if there is only one DOF
    if num_plots == 1: axes = [axes]
        
    for i in range(num_plots):
        r, c = rows[i], cols[i]
        axes[i].plot(t_values, cov_matrices[:, r, c])
        axes[i].set_title(f"Degree of Freedom: ({r}, {c})")
        axes[i].grid(True)
        
    plt.savefig(filename)
    plt.close()
    print(f"Saved all components to: {filename}")