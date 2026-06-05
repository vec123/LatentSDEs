import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm

class CovarianceModel:
    def __init__(self, D, type='full'):
        self.D = D
        self.type = type

    def get_cov(self, **kwargs):
        raise NotImplementedError("Subclasses must implement get_cov")
    
    
    def get_dot_cov(self, **kwargs):
        raise NotImplementedError("Subclasses must implement get_dot_cov")