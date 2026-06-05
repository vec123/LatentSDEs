import jax
import numpy as np
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
from src.sde.GaussianPaths.means.matern_52_mean import MaternMeanModel
from src.sde.GaussianPaths.Variances.matern_52_var import MaternCovariance
from src.sde.tests.obstacles import get_obstacle_loss
from src.sde.tests.plot_helper import plot_and_save_potential
from src.sde.tests.plot_helper import plot_model_path, plot_knots_and_coeffs, plot_static_path_visualization


def fit_and_plot():
    NUM_EVAL = 100
    T = 20.0
    eps = 1e-6
    t_support = jnp.linspace(0, T, 50)

    start_pos, end_pos = jnp.array([-1.0, -1.0]), jnp.array([1.0, 1.0])
    obstacles = [{'center': (0.0, 0.1), 'radius': 0.4}]
    plot_and_save_potential(obstacles)

    # Parameters to optimize
    NUM_SUPPORT = 50 
    
    # Init Hm as a straight line between start and end
    Hm_init = start_pos + (end_pos - start_pos) * (jnp.linspace(0, 1, NUM_SUPPORT)[:, None])
    Hm_opt = jnp.copy(Hm_init[1:-1]) 
    print(f"Initial Hm shape: {Hm_init.shape}")  # Should be (50, 2)
    print(f"Optimizing Hm shape: {Hm_opt.shape}")  # Should be (48, 2)
    params_R = jnp.zeros((1, 51)) 
    params_L = jnp.zeros((2, 51))
    
    optimizer = optax.adam(0.01)
    params = {
        "path_anchors" = Hm_opt,
        "covariance_R" = params_R,
        "covariance_L" = params_L,
    }
    opt_state = optimizer.init(params)

    def loss_fn(params):
            
            curr_HM_opt = params["path_anchors"]
            params_R, params_L = params["covariance_R"],params[ "covariance_L"]
            
            # Reconstruct full Hm
            curr_Hm = jnp.concatenate([start_pos[None, :], curr_HM_opt, end_pos[None, :]], axis=0)  
            mean_model = MaternMeanModel(t_support, curr_Hm, length_scale=0.1, sigma_f=1.0, T=T)

            # Evaluate the path at the evaluation grid 
            path = jax.vmap(mean_model)(t_eval)
            dot_path = jax.vmap(mean_model.get_dot_mu)(t_eval)


            loss = 0.0
            #Boundary Check
            print(f"Start Pos: {start_pos}, End Pos: {end_pos}")
            print(f"start path: {path[0]}, end path: {path[-1]}")
            print(f"curr_Hm[0]: {curr_Hm[0]}, curr_Hm[-1]: {curr_Hm[-1]}")

            # Obstacle Penalty
            print(f"Evaluating Obstacle Loss")
            #loss += jnp.mean(jax.vmap(lambda m: get_obstacle_loss(m, obstacles))(path))
            obstacle_penalty = jnp.mean(jax.vmap(lambda m: get_obstacle_loss(m, obstacles))(path))
            loss += 1 * obstacle_penalty # Start with a small scale
            print(f"Obstacle Penalty: {1 * obstacle_penalty}")

            # Kinetic Loss 
            print(f"Evaluating kinetic Loss")
            loss +=1* jnp.mean(dot_path**2) 
            print(f"Kinetic Loss: {1 * jnp.mean(dot_path**2)}")
            return loss

    # Optimization
    for i in range(100):
        key = jax.random.PRNGKey(i)
        t_eval = jnp.sort(jax.random.uniform(key, shape=(NUM_EVAL,), minval=eps, maxval=T-eps))

        loss, grads = jax.value_and_grad(loss_fn)((Hm_opt, params_R, params_L))
        updates, opt_state = optimizer.update(grads, opt_state)
        Hm_opt, params_R, params_L = optax.apply_updates((Hm_opt, params_R, params_L), updates)
        if i % 10 == 0:
         print(f"Iteration {i}, Loss: {loss}")

        Hm_final = jnp.concatenate([start_pos[None, :], Hm_opt, end_pos[None, :]], axis=0)
        mean_model = MaternMeanModel(t_support, Hm_final, length_scale=0.1, sigma_f=1.0, T=T)
        plot_model_path(t_eval, mean_model, file_name=f"paths/model_path_{i}.png")

if __name__ == "__main__":
    fit_and_plot()