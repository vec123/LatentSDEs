import jax
import jax.numpy as jnp
from jax.test_util import check_grads
# Assuming your file is named matern_52_var.py
from src.sde.GaussianPaths.Variances.matern_52_var import MaternCovariance

from jax import config
config.update("jax_enable_x64", True)

def run_tests():
    # 1. Initialization
    D = 2
    T = 10.0
    model = MaternCovariance(D=D, time_interval=(0.0, T), num_basis=50, type='full')
    
    key = jax.random.PRNGKey(42)
    weights_R = jax.random.normal(key, (model.dim_R, 51))
    weights_L = jax.random.normal(key, (D, 51))
    
    # Common hyperparameters
    args = (0.5, weights_R, weights_L, 1.0, 1.0, 2.0, 1.0)
    
    print("--- Running Validity Tests ---")
    
    # 2. Test Symmetry and Positive Definiteness
    S = model.get_cov(*args)
    assert jnp.allclose(S, S.T, atol=1e-5), "S is not symmetric!"
    eigvals = jnp.linalg.eigvals(S)
    assert jnp.all(eigvals > 0), f"S is not Positive Definite! Eigs: {eigvals}"
    print("✓ Covariance is Symmetric and Positive Definite.")
    
    # 3. Test Differentiability of get_cov
    # Check if we can compute gradients through the covariance model
    def loss_func(params):
        W_R, W_L = params
        S = model.get_cov(0.5, W_R, W_L, 1.0, 1.0, 2.0, 1.0)
        return jnp.sum(S**2)
    
    params = (weights_R, weights_L)
    check_grads(loss_func, (params,), order=1, modes=['rev'], eps=1e-3, atol=1e-2, rtol=1e-2)
    print("✓ Autodiff through get_cov is working.")
    
    # 4. Test Jacobian derivative consistency
    # get_dot_cov uses jax.jacobian. Verify it returns a (D,D) matrix
    dot_S = model.get_dot_cov(*args)
    assert dot_S.shape == (D, D), f"Expected shape ({D},{D}), got {dot_S.shape}"
    print("✓ get_dot_cov returns valid matrix shape.")

    print("\nAll tests passed successfully!")

if __name__ == "__main__":
    run_tests()