import jax
import optax
import jax.numpy as jnp
from jax.test_util import check_grads
from src.sde.GaussianPaths.Variances.b_spline_var import BSplineCovariance
from src.sde.tests.gp_ground_truths import get_ground_truth_cov
from src.sde.tests.plot_helper import plot_covar_components_dxd

from jax import config
config.update("jax_enable_x64", True)

def run_tests():
    #  Initialization
    D = 2
    T = 10.0
    eps = 1e-6
    NUM_TIMES = 500
    NUM_BASIS = 10
    DEGREE = 3
    t_array = jnp.linspace(0, T, NUM_TIMES)
    S_true = jax.vmap(get_ground_truth_cov)(t_array)
    plot_covar_components_dxd(t_array, S_true, "test_cov.png")

    knots = jnp.linspace(0.0, T, NUM_BASIS)
    model = BSplineCovariance(D=D,  knots=knots, degree =DEGREE,  t=t_array, S_true=S_true, type="full")
    S_pred = jax.vmap(
                model.get_cov, 
                in_axes=(0, None, None,None)
            )(t_array, model.W_R, model.W_L, knots)
    plot_covar_components_dxd(t_array, S_pred, f"pred_cov_init.png")


    params_R = model.W_R
    params_L = model.W_L
    knot_diffs = jnp.diff(knots)
    params = {
        "params_R":params_R,
        "params_L":params_L,
     #   "knots": knots
     #   "knot_diffs":knot_diffs
    }

    optimizer = optax.adam(0.1)
    opt_state = optimizer.init(params)

    def loss_fn(params):
        W_R, W_L = params["params_R"], params["params_L"]
      #  knots =  params["knots"]
      #  knot_diffs = jax.nn.softplus(params["knot_diffs"]) + 1e-6
      #  knots = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(knot_diffs)])
        S_pred = jax.vmap(
                model.get_cov, 
                in_axes=(0, None, None,None)
            )(t_array, W_R, W_L, knots)
        loss = jnp.sum( (S_pred - S_true)**2)
        reg = jnp.mean(jnp.diff(W_L, n=2, axis=1)**2) + jnp.mean(jnp.diff(W_R, n=2, axis=1)**2)
        return loss + 1e-1 *reg 
    
    @jax.jit
    def step(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    
    for i in range(201): 
        key = jax.random.PRNGKey(i)
        params, opt_state, loss = step(params, opt_state)
        if i % 100 == 0: 
            print(f" Epoch {i}, Loss: {loss:.4f}")
            W_R, W_L = params["params_R"], params["params_L"]
       #     knots =  params["knots"]
        #    knot_diffs = jax.nn.softplus(params["knot_diffs"]) + 1e-6
        #    knots = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(knot_diffs)])
            print("knots: ", knots)
            S_pred = jax.vmap(
                model.get_cov, 
                in_axes=(0, None, None,None)
            )(t_array, W_R, W_L, knots)
            loss = jnp.mean( (S_pred - S_true)**2)
            plot_covar_components_dxd(t_array, S_pred, f"pred_cov_{i}.png")
            plot_covar_components_dxd(t_array, S_pred-S_true, f"error_{i}.png")

            #param_str = " | ".join([f"{k}: {v.shape if v.ndim > 0 else v:.4f}" for k, v in params.items()])


if __name__ == "__main__":
    run_tests()