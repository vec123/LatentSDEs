import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
from src.models.b_splines import JAXBSpline 
from src.sde.GaussianPaths.means.b_spline_mean import BSplineMeanModel
from src.sde.tests.obstacles import get_obstacle_loss
from src.sde.tests.plot_helper import plot_and_save_potential
from src.sde.tests.plot_helper import plot_model_path, plot_knots_and_coeffs, plot_static_path_visualization

def fit_and_plot():
    T = 20.0

    degree = 3
    NUM_KNOTS = 10
    NUM_SUPPORT = 100
    NUM_EVAL = 100
    dt = T/NUM_EVAL 
    t_support= jnp.linspace(0, T, NUM_SUPPORT)
    eps = 1e-6
    t_eval = jnp.linspace(0, T-eps, NUM_EVAL)

    # Enpoints
    start_pos, end_pos = jnp.array([-1.5, -1.0]), jnp.array([1.5, 1.0])

    # Initialize á linear trajectory as a straight line 
    y_init = start_pos + (end_pos - start_pos) * (jnp.linspace(0, 1, NUM_SUPPORT)[:, None])
    print("y_init.shape: ", y_init.shape)
    #fit the BSpline model to this line

    knots = jnp.linspace(t_support.min(), t_support.max(), NUM_KNOTS)
    model = BSplineMeanModel(t = t_support, knots= knots, y = y_init, degree=degree)
    path = jax.vmap(model)(t_eval)
    print(f"init start path: {path[0]}, init end path: {path[-1]}")
    coeffs_init = model.coeffs
    inner_coeffs_init = coeffs_init[1:-1]  # Only optimize inner coefficients, keeps the endpoints fixed
    full_coeffs_init = jnp.concatenate([
        start_pos[None, :],   # Fixed
        inner_coeffs_init,     # Optimized
        end_pos[None, :]      # Fixed
    ], axis=0)

    model = BSplineMeanModel(t = t_support, knots= knots, coeffs = full_coeffs_init, degree=degree)
    plot_model_path(t_eval, model, file_name="paths/b_spline_model_path_init.png")
    plot_knots_and_coeffs(model,  file_name=f"paths/b_spline_model_knots_coeffs_init.png")

    obstacles = [{'center': (0.0, 0.1), 'radius': 0.4}]
    plot_and_save_potential(obstacles, file_name="paths/b_spline_potential_field.png")
    optimizer = optax.adam(learning_rate=0.1)
    params = {
    'coeffs': inner_coeffs_init,
    'knots': knots
    }
    opt_state = optimizer.init(params)

    #  Loss Function
    def loss_fn(params):
        curr_coeffs_opt = params['coeffs']
        knots_opt = jnp.sort(jnp.clip(params['knots'], 0.0, T))
                              
        full_coeffs = jnp.concatenate([
            start_pos[None, :],   # Fixed
            curr_coeffs_opt,         # Optimized
            end_pos[None, :]      # Fixed
        ], axis=0)
               
        model = BSplineMeanModel(t = t_support, knots = knots_opt , coeffs = full_coeffs, degree=degree)

        path = jax.vmap(model)(t_eval)
        dot_path = jax.vmap(model.get_dot_mu)(t_eval)
        # Obstacle Penalty
        obstacle_penalty = jnp.mean(jax.vmap(lambda p: get_obstacle_loss(p, obstacles))(path))
        
        loss = 0.0
        # 1. Boundary Constraints
        #print(f"Evaluating Boundary Loss: Start {path[0]}, End {path[-1]}")
        print(f"Start Pos: {start_pos}, End Pos: {end_pos}")
        print(f"start path: {path[0]}, end path: {path[-1]}")
        #loss = 100*jnp.mean((path[0] - start_pos)**2) + jnp.mean((path[-1] - end_pos)**2)
        #print(f"Boundary Loss: {loss}")

        # 2. Obstacle Penalty
        print(f"Evaluating Obstacle Loss")
        obstacle_penalty = jnp.mean(jax.vmap(lambda m: get_obstacle_loss(m, obstacles))(path))
        loss += 1 * obstacle_penalty # Start with a small scale
        print(f"Obstacle Penalty: {1 * obstacle_penalty}")

        # 3. Regularization (Trace of cov + Smoothness)
        #loss += 0.01 * jnp.mean(jnp.trace(covs, axis1=1, axis2=2))
        print(f"Evaluating kinetic Loss")
        loss += 500* jnp.mean(dot_path**2) 
        print(f"Kinetic Loss: {500 * jnp.mean(dot_path**2)}")

        #print(f"Evaluating acc Loss")
        #loss += 0.0000000001* jnp.mean(ddot_path**2) 
        #print(f"Acc Loss: {0.0000001* jnp.mean(ddot_path**2)}")

        return loss

    # 4. Training Loop
    curr_coeffs_opt = inner_coeffs_init
    for i in range(10):
        key = jax.random.PRNGKey(i)

        t_eval = jnp.sort(jax.random.uniform(key, shape=(NUM_EVAL,), minval=eps, maxval=T-eps))
        #base_t = jnp.linspace(0, T, NUM_EVAL)
        #jitter = jax.random.uniform(key, shape=(NUM_EVAL,), minval=-dt/2, maxval=dt/2)
        #t_eval = jnp.clip(base_t + jitter, eps, T-eps)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        if i % 20 == 0:
            print(f"Iteration {i}, Loss: {loss:.4f}")
        
        curr_coeffs_opt = params['coeffs']
        knots_opt = params['knots']
        curr_coeffs = jnp.concatenate([
            start_pos[None, :],   # Fixed
            curr_coeffs_opt,         # Optimized
            end_pos[None, :]      # Fixed
        ], axis=0)
        model = BSplineMeanModel(t = t_support,knots = knots_opt, coeffs = curr_coeffs, degree=degree)
        plot_model_path(t_eval, model, file_name=f"paths/b_spline_model_path_{i}.png")
        plot_knots_and_coeffs(model,  file_name=f"paths/b_spline_model_knots_coeffs_{i}.png")
    
    # 5. Final Visualization
    curr_coeffs_opt = params['coeffs']
    knots_opt = params['knots']
    final_coeffs = jnp.concatenate([
        start_pos[None, :],   # Fixed
        curr_coeffs_opt,         # Optimized
        end_pos[None, :]      # Fixed
    ], axis=0)

    final_model = BSplineMeanModel(t = t_support, knots = knots_opt, coeffs = final_coeffs, degree=degree)
    plot_model_path(t_eval, final_model, file_name="paths/b_spline_model_path_final.png")
    plot_static_path_visualization(final_model, obstacles, T-eps, file_name="paths/final_path.png")

if __name__ == "__main__":
    fit_and_plot()