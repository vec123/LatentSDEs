import jax
import optax
import jax.numpy as jnp
from jax.test_util import check_grads
from src.sde.GaussianPaths.Variances.matern_52_var import MaternCovariance
from src.sde.tests.gp_ground_truths import get_ground_truth_cov
from src.sde.tests.plot_helper import plot_covar_components_dxd

from jax import config
config.update("jax_enable_x64", True)

def fit_and_plot():
    #Initialization
    D = 2
    T = 10.0
    NUM_TIMES = 100
    NUM_BASIS = 20
    length_scale = 1
    alpha = 1.0
    beta = 1.0
    _sigma_ = 1.0
    basis_centers = jnp.linspace(0, T, NUM_BASIS)

    t_array = jnp.linspace(0, T, NUM_TIMES)
    S_true = jax.vmap(get_ground_truth_cov)(t_array)
    plot_covar_components_dxd(t_array, S_true, "test_cov.png")
     
    model = MaternCovariance(D=D, time_interval=(0.0, T), type="full")
    params_R = jax.random.normal(jax.random.PRNGKey(0), (model.dim_R, NUM_BASIS + 1)) * 1e-2
    params_L = jax.random.normal(jax.random.PRNGKey(1), (D, NUM_BASIS + 1)) * 0.1
    params = {
        "params_R":params_R,
        "params_L":params_L,
        "basis_centers": basis_centers,
        "logit_length_scale": jnp.array(0.0),
       # "log_alpha": jnp.array(0.0),
      #  "log_beta": jnp.array(0.0),
    }

    optimizer = optax.adam(0.001)
    opt_state = optimizer.init(params)

    def get_alpha_beta_params(params):
        alpha = 0.1 + 4.9 * jax.nn.sigmoid(params["log_alpha"])
        beta = 0.1 + 4.9 * jax.nn.softplus(params["log_beta"])
        return alpha, beta
    
    def loss_fn(params):
        W_R, W_L = params["params_R"], params["params_L"]
        basis_centers = params["basis_centers"]
      #  alpha,beta =  get_alpha_beta_params(params)
        length_scale = jax.nn.sigmoid(params["logit_length_scale"])
        S_pred = jax.vmap(
            model.get_cov, 
            in_axes=(0, None, None, None, None, None, None, None)
        )(t_array, W_R, W_L, basis_centers, length_scale,alpha, beta, _sigma_)
        loss = jnp.sum( (S_pred - S_true)**2)
        return loss
    
    @jax.jit
    def step(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    
    for i in range(2001): 
        params, opt_state, loss = step(params, opt_state)
        
        if i % 100 == 0: 
            print(f"Epoch {i}, Loss: {loss:.4f}")
            W_R, W_L = params["params_R"], params["params_L"]
            basis_centers = params["basis_centers"]
            #alpha,beta =  get_alpha_beta_params(params)
            length_scale = jax.nn.sigmoid(params["logit_length_scale"])
            S_pred = jax.vmap(
                model.get_cov, 
                in_axes=(0, None, None, None, None, None, None, None)
            )(t_array, W_R, W_L, basis_centers, length_scale, alpha, beta, _sigma_)
            loss = jnp.mean( (S_pred - S_true)**2)
            plot_covar_components_dxd(t_array, S_pred, f"pred_cov_{i}.png")
            plot_covar_components_dxd(t_array, S_pred-S_true, f"error_{i}.png")
           
            param_str = " | ".join([f"{k}: {v.shape if v.ndim > 0 else v:.4f}" for k, v in params.items()])
            print(f"Epoch {i} | Loss: {loss:.4f} | {param_str}")

if __name__ == "__main__":
    fit_and_plot()