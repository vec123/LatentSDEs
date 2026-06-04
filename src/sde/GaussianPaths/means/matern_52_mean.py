import jax
import jax.numpy as jnp
import haiku as hk
from scipy.interpolate import make_lsq_spline, BSpline

from src.models.matern_kernels import matern_52_kernel
from src.sde.GaussianPaths.means.base import BaseMeanModel

class MaternMeanModel(BaseMeanModel):
    def __init__(self, time_points, Hm, length_scale, sigma_f, noise_var=1e-6):
        self.time_points = time_points
        self.Hm = Hm
        self.length_scale = length_scale
        self.sigma_f = sigma_f
        
        # Precompute the weights for the GP interpolation
        # K_MM: (N, N)
        K_MM = matern_52_kernel(time_points, time_points, length_scale, sigma_f**2)
        # Solve (K + sigma*I) * W = Hm
        self.weights = jax.scipy.linalg.solve(
            K_MM + noise_var * jnp.eye(time_points.shape[0]), 
            Hm, 
            assume_a='pos'
        )
        
    def __call__(self, t):
        # K(t, time_points): (1, N)
        kt = matern_52_kernel(t, self.time_points, self.length_scale, self.sigma_f**2)
        return jnp.dot(kt, self.weights)
    

