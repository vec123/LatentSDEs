import jax.numpy as jnp

class CovarianceModel:
    def __init__(self, dim, type='full'):
        self.dim = dim
        self.type = type

    def get_cov(self, params):
        if self.type == 'diagonal':
            # params is just log-variances
            return jnp.diag(jnp.exp(params))
        
        elif self.type == 'full':
            # params contains elements of L and diagonal D
            # Construct L (lower triangular with 1s on diagonal)
            L = jnp.tril(params['L'], k=-1) + jnp.eye(self.dim)
            D = jnp.diag(jnp.exp(params['log_d']))
            return L @ D @ L.T