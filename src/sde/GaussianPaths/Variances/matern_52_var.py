import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm
from src.models.matern_kernels import kumaraswamy_warping, matern_52_kernel
from src.sde.GaussianPaths.Variances.base import CovarianceModel

class MaternCovariance(CovarianceModel):
    def __init__(self, D, time_interval, length_scale = 0.5, type='full', num_basis=200):
        super().__init__(D, type)
        self.T = time_interval[1]
        self.basis_centers = jnp.linspace(0, self.T, num_basis)
        self.dim_R = D * (D - 1) // 2
        self.dim_Lambda = D
        self.length_scale = length_scale

    def get_basis_vector(self, t, alpha, beta, sigma):
        t_warped = kumaraswamy_warping(t, self.T, alpha, beta)
        centers_warped = kumaraswamy_warping(self.basis_centers, self.T, alpha, beta)
        k_vals = jax.vmap(lambda c: matern_52_kernel(t_warped, c, self.length_scale, sigma))(centers_warped)
        return jnp.concatenate([jnp.ones((1,)), k_vals])

    def _to_skew_symmetric(self, vec):
        mat = jnp.zeros((self.D, self.D))
        i_idx, j_idx = jnp.triu_indices(self.D, k=1)
        mat = mat.at[i_idx, j_idx].set(vec)
        mat = mat.at[j_idx, i_idx].set(-vec)
        return mat

    def get_cov(self, t, weights_R, weights_Lambda, alpha, beta, sigma):
        phi = self.get_basis_vector(t, alpha, beta,  sigma)
        vec_Lambda = jnp.dot(weights_Lambda, phi)
        Lambda_phi = jnp.diag(jax.nn.softplus(vec_Lambda))
        
        if self.type == 'diagonal' or self.D == 1:
            return Lambda_phi + 1e-6 * jnp.eye(self.D)
        
        vec_R = jnp.dot(weights_R, phi)
        R_phi = expm(self._to_skew_symmetric(vec_R))
        assert R_phi.shape == (self.D, self.D)
        assert Lambda_phi.shape == (self.D, self.D)
        
        return R_phi @ Lambda_phi @ R_phi.T + 1e-6 * jnp.eye(self.D)
      
    def get_dot_cov(self, t, weights_R, weights_Lambda, alpha, beta, sigma):
        def cov_at_t(t_in):
            phi = self.get_basis_vector(t_in, alpha, beta, sigma)
            vec_Lambda = jnp.dot(weights_Lambda, phi)
            Lambda = jnp.diag(jax.nn.softplus(vec_Lambda))
            
            # Add the same guard here
            if self.D == 1 or self.type == 'diagonal':
                return Lambda
                
            vec_R = jnp.dot(weights_R, phi)
            omega = self._to_skew_symmetric(vec_R)
            R = expm(omega)
            return R @ Lambda @ R.T

        return jax.jacobian(cov_at_t)(t)
    """
    def get_dot_diagonal_only(self, t, weights_R, weights_Lambda, alpha, beta, sigma):
       
        #Returns R * dot_Lambda * R^T. 
        #Requires weights_R to construct R.
        
        phi = self.get_basis_vector(t, alpha, beta, sigma)
        d_phi_dt = jax.grad(lambda t_in: self.get_basis_vector(t_in, alpha, beta, sigma))(t)
        
        vec_Lambda = jnp.dot(weights_Lambda, phi)
        d_vec_Lambda = jnp.dot(weights_Lambda, d_phi_dt)
        dot_Lambda = jnp.diag(jax.nn.sigmoid(vec_Lambda) * d_vec_Lambda)
        
        # Construct R_phi
        vec_R = jnp.dot(weights_R, phi)
        R_phi = expm(self._to_skew_symmetric(vec_R))
        
        return R_phi @ dot_Lambda @ R_phi.T
     """