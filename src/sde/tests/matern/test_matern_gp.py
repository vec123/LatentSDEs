import jax
import jax.numpy as jnp
import optax
from jax.lax.linalg import triangular_solve

from src.sde.GaussianPaths.means.matern_52_mean import MaternMeanModel
from src.sde.GaussianPaths.Variances.matern_52_var import MaternCovariance
from src.sde.tests.gp_ground_truths import sample_trajectories, plot_gp_comparison
from src.sde.tests.gp_ground_truths import plot_gp_ground_truth, plot_empirical_vs_true
from src.sde.tests.data_helper import generate_data

from jax import config
config.update("jax_enable_x64", True)

def fit_models(use_covar=True):
    # 1. Generate Data
    T = 20.0
    use_gp_ground_truth = True
    if use_gp_ground_truth:
        t_vals = jnp.linspace(0, T, 80)
        trajs = sample_trajectories(t_vals, num_trajs=100)
        all_t = jnp.concatenate([t for t, y in trajs])
        all_y = jnp.concatenate([y for t, y in trajs]) # Expected shape: (N, D)
        #plot_gp_ground_truth(trajs, t_vals, dim=all_y.shape[1] if all_y.ndim > 1 else 1)
    else:
        # 1. Generate Ensemble
        trajs, t_vals = generate_data(num_trajectories=10, num_points=100, T=T)
        all_t = jnp.concatenate([t for t, y in trajs])
        all_y = jnp.concatenate([y for t, y in trajs])
        all_y = all_y.reshape(-1, 1) # Ensure shape is (N, D) even for 1D data
        print(f"Generated data with shape: {all_y.shape}")

    dim = all_y.shape[1] if all_y.ndim > 1 else 1
    print(f"Fitting GP for dimension: {dim}")

    # 2. Fit Mean Model
    mean_model = MaternMeanModel(all_t, all_y, length_scale=0.5, sigma_f=1.0, T=T, alpha=1.0, beta=1.0)
    
    # 3. Handle Covariance
    num_basis = 400
    cov_model = MaternCovariance(D=dim, time_interval=(0.0, T), length_scale= 1, num_basis=num_basis, type="diagonal")
    
    # Initialize parameters
    params_R = jax.random.normal(jax.random.PRNGKey(0), (cov_model.dim_R, num_basis + 1)) * 1e-2
    params_L = jax.random.normal(jax.random.PRNGKey(1), (dim, num_basis + 1)) * 0.1 

    optimizer = optax.adam(1e-2)
    opt_state = optimizer.init((params_R, params_L))
    
    if use_covar:
        mu_pred = jax.vmap(mean_model)(all_t)
        residuals = (all_y - mu_pred)

        def loss_fn(params):
                    W_R, W_L = params
                    
                    # 1. Ensure S_all is (N, D, D)
                    S_all = jax.vmap(lambda t: cov_model.get_cov(t, W_R, W_L, 1.0, 2.0, 1.0))(all_t)
                    
                    # 2. Ensure residuals is (N, D, 1) to act as column vectors
                    # This makes r=(D, 1), which fits the solve(a=(D,D), b=(D,1)) requirement
                    residuals_col = residuals[:, :, jnp.newaxis] 
                    
                    def nll_per_point(S, r):
                        # S is (D, D), r is (D, 1)
                        inv_S_r = jax.scipy.linalg.solve(S, r, assume_a='pos')
                        quad = jnp.dot(r.T, inv_S_r)
                        
                        L = jax.scipy.linalg.cholesky(S, lower=True)
                        log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
                        
                        return log_det + quad.squeeze() # Squeeze to return scalar

                    # Now vmap maps over the first dimension (N) of both S_all and residuals_col
                    nll_values = jax.vmap(nll_per_point)(S_all, residuals_col)
                    return jnp.mean(nll_values)

        for i in range(20):
            loss, grads = jax.value_and_grad(loss_fn)((params_R, params_L))
            updates, opt_state = optimizer.update(grads, opt_state)
            params_R, params_L = optax.apply_updates((params_R, params_L), updates)
            if i % 20 == 0: print(f"Epoch {i}, Loss: {loss:.4f}")

    plot_gp_comparison(mean_model, cov_model, t_vals, params_R, params_L, trajs=None, dim=dim)
    plot_empirical_vs_true(trajs, t_vals, params_R, params_L, mean_model, cov_model)
if __name__ == "__main__":
    fit_models()