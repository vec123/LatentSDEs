import jax
import jax.numpy as jnp
from jax.nn import softplus

def init_gaussian_params(key, dim, type='full'):
    # Mean: initialized to zero
    mu = jnp.zeros(dim)
    
    if type == 'diagonal':
        # Log-variance (exp will ensure positivity)
        log_var = jnp.zeros(dim)
        return {'mu': mu, 'log_var': log_var}
    else:
        # Full covariance via Cholesky L: initialize as Identity
        L = jnp.eye(dim)
        return {'mu': mu, 'L': L}

def sample_gaussian(key, params, type='full'):
    mu = params['mu']
    if type == 'diagonal':
        std = jnp.exp(0.5 * params['log_var'])
        noise = jax.random.normal(key, shape=mu.shape)
        return mu + std * noise
    else:
        L = params['L']
        # Ensure L is lower triangular
        L = jnp.tril(L)
        # Force diagonal elements to be positive for valid Cholesky
        L = L.at[jnp.diag_indices_from(L)].set(softplus(jnp.diag(L)))
        noise = jax.random.normal(key, shape=mu.shape)
        return mu + L @ noise