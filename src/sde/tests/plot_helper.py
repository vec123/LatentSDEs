import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

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

def plot_ensemble(model, trajs, title, filename):
    plt.figure(figsize=(10, 6))
    
    # Plot all trajectories
    for t, y in trajs:
        plt.plot(t, y, 'gray', alpha=0.3, label='Ground Truth' if t is trajs[0][0] else "")
    
    # Plot mean
    t_dense = jnp.linspace(trajs[0][0].min(), trajs[0][0].max(), 100)
    # Using list comprehension to bypass vmap/tracer issues for generic model types
    y_pred = jnp.array([model(ti) for ti in t_dense])
    
    plt.plot(t_dense, y_pred, 'r-', linewidth=2, label='Model Mean Prediction')
    plt.title(title)
    plt.legend()
    plt.savefig(filename)
    plt.close()