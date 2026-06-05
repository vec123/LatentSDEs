import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm

from src.models.b_splines import JAXBSpline
from src.sde.GaussianPaths.Variances.base import CovarianceModel

class BSplineCovariance(CovarianceModel):
    def __init__(self, D, knots, degree=3, type='full'):
        super().__init__(D, type)
        # Use your JAXBSpline class defined earlier
        self.spline_basis = JAXBSpline(knots, degree)
        
    def get_basis_vector(self, t):
        # Returns vector of basis functions at time t
        return self.spline_basis.get_design_matrix(t).flatten()

    def get_cov(self, t, W_R, W_Lambda):
        phi = self.get_basis_vector(t)
        
        # Lambda construction
        vec_Lambda = jnp.dot(W_Lambda, phi)
        Lambda = jnp.diag(jax.nn.softplus(vec_Lambda))
        
        if self.type == 'diagonal':
            return Lambda
        
        # Rotation construction
        vec_R = jnp.dot(W_R, phi)
        omega = self._to_skew_symmetric(vec_R)
        R = expm(omega)
        return R @ Lambda @ R.T