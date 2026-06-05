import jax
import numpy as np
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt5r4
from src.sde.GaussianPaths.means.nn_mean import NNMeanModel
from src.sde.GaussianPaths.Variances.nn_var import NNCovariance
from src.sde.tests.obstacles import get_obstacle_loss
from src.sde.tests.plot_helper import plot_and_save_potential
from src.sde.tests.plot_helper import plot_model_path, plot_knots_and_coeffs, plot_static_path_visualization


def prefit(path_data, model, epochs=500, learning_rate=0.01):
    """
    path_data: (y_init, t_support) - the target path points and their time indices
    model: An instance of a NNModel
    """
    y_target, t_support = path_data
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(model.params)

    # MSE Loss for prefitting
    def loss_fn(params, t_array, target_array):
        #prediction = model.apply_fn(params, None, t)
        path = jax.vmap(lambda t: model.apply_fn(params, None, t))(t_array)
        #path = jax.vmap(model.get_mu)(t_array)
        dot_path = jax.vmap(model.get_dot_mu)(t_array)
        
        recon = jnp.mean((path - target_array)**2)
        kinetic = 10*jnp.mean(dot_path**2) 
        loss =  recon  + kinetic
        return loss

    # JIT-compiled update step for speed
    @jax.jit
    def step(params, opt_state, t, target):
        loss, grads = jax.value_and_grad(loss_fn)(params, t, target)
        updates, opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, loss

    print("Starting prefit...")
    params = model.params
    for epoch in range(epochs):
        params, opt_state, loss = step(params, opt_state, t_support, y_target)
        if epoch % 100 == 0:
            print(f"Prefit Epoch {epoch}, Loss: {loss:.6f}")
    
    # Update the model's internal params with the fitted weights
    model.params = params
    print("Prefit complete.")
    return model

def fit_and_plot():
    NUM_EVAL = 100
    T = 20.0
    eps = 1e-6
    

    start_pos, end_pos = jnp.array([-1.0, -1.0]), jnp.array([1.0, 1.0])

    obstacles = [{'center': (0.0, 0.1), 'radius': 0.4}]
    plot_and_save_potential(obstacles)

    # Parameters to optimize
    NUM_SUPPORT = 50 
    t_support = jnp.linspace(0, T, NUM_SUPPORT)
    y_init = start_pos + (end_pos - start_pos) * (jnp.linspace(0, 1, NUM_SUPPORT)[:, None])

    model = NNMeanModel(in_dim=1, out_dim=2, hidden_sizes=[32, 32])
    model = prefit((y_init, t_support), model)

    t_eval =  t_support = jnp.linspace(0, T, NUM_EVAL)
    plot_model_path(t_eval, model, file_name=f"paths/model_path_init.png")
    
    # Use the initial parameters from prefit
    current_params = model.params
    optimizer = optax.adam(0.01)
    opt_state = optimizer.init(current_params)

    # Pure loss function: accepts params, returns loss
    def loss_fn(params):
        model.params = params
        path = jax.vmap(model)(t_eval)
        dot_path = jax.vmap(model.get_dot_mu)(t_eval)
        
        loss = 0
        print(f"Start Pos: {start_pos}, End Pos: {end_pos}")
        print(f"start path: {path[0]}, end path: {path[-1]}")
        boundary_loss= jnp.mean( (start_pos - path[0])**2 + (end_pos -path[-1])**2 )
        loss += boundary_loss
        
        # 4. Losses
        obstacle_penalty = jnp.mean(jax.vmap(lambda m: get_obstacle_loss(m, obstacles))(path))
        print(f"Obstacle_penalty : ", obstacle_penalty )
        kinetic_loss = 0.01*jnp.mean(dot_path**2)
        
        print(f"kinetic_loss : ", kinetic_loss )
        loss = boundary_loss + obstacle_penalty + kinetic_loss
        return loss

    # Optimization Loop
    for i in range(1000):
        key = jax.random.PRNGKey(i)
        t_eval = jnp.sort(jax.random.uniform(key, shape=(NUM_EVAL,), minval=eps, maxval=T-eps))
        
        # value_and_grad now operates on 'current_params'
        loss, grads = jax.value_and_grad(loss_fn)(current_params, t_eval)
        
        updates, opt_state = optimizer.update(grads, opt_state)
        current_params = optax.apply_updates(current_params, updates)

        # Update the model object ONLY when you need to visualize or save
        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss}")
            model.params = current_params 
            plot_model_path(t_eval, model, file_name=f"paths/model_path_{i}.png")


if __name__ == "__main__":
    fit_and_plot()