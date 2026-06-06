import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm
from src.models.b_splines import JAXBSpline
from src.sde.GaussianPaths.Variances.base import CovarianceModel

class BSplineCovariance(CovarianceModel):
    def __init__(self, D, knots, degree=3, t =None, S_true=None, W_L = None, W_R = None, type='full'):
        super().__init__(D, type)
        
        self.degree = degree
        self.knots = jnp.sort(knots)
        self.knots_padded = self.knots 
        self.knots_padded = self.pad_knots(t,self.degree,knots)
 

        self.dim_R = D * (D - 1) // 2
        self.dim_Lambda = D
       
        self.spline_basis = JAXBSpline(self.knots_padded, degree)
        self.num_basis = self.spline_basis.get_design_matrix(jnp.array([t[0]])).shape[1]

        if S_true is not None and t is not None:
            self.W_L, self.W_R = self.fit_initial_covariance_weights(t, S_true, self.spline_basis, D)

    def pad_knots(self, t, degree, knots):
        # Calculate average spacing to extrapolate
        spacing = jnp.mean(jnp.diff(knots))
        
        # Create padding by extending the knot sequence linearly
        # This avoids the "clamping" effect of repeating t.min()
        left_pad = knots[0] - spacing * jnp.arange(degree + 1, 0, -1)
        right_pad = knots[-1] + spacing * jnp.arange(1, degree + 2)
        return jnp.concatenate([left_pad, knots, right_pad])
    
    def _to_skew_symmetric(self, vec):
        """Generalized skew-symmetric construction for any D."""
        mat = jnp.zeros((self.D, self.D))
        i_idx, j_idx = jnp.triu_indices(self.D, k=1)
        mat = mat.at[i_idx, j_idx].set(vec)
        mat = mat.at[j_idx, i_idx].set(-vec)
        return mat

    def get_cov(self, t, weights_R, weights_Lambda, knots):
       
        #knots_padded = knots
        knots_padded = self.pad_knots(t,self.degree,knots)
        
        spline_basis = JAXBSpline(knots_padded, self.degree)
        phi = spline_basis.get_design_matrix(t).flatten()
        
        # Lambda construction
        vec_Lambda = jnp.dot(weights_Lambda, phi)
        Lambda_phi = jnp.diag(jax.nn.softplus(vec_Lambda))
        
        if self.type == 'diagonal' or self.D == 1:
            return Lambda_phi + 1e-6 * jnp.eye(self.D)
        
        # Rotation construction
        vec_R = jnp.dot(weights_R, phi)
        R_phi = expm(self._to_skew_symmetric(vec_R))
        
        return R_phi @ Lambda_phi @ R_phi.T + 1e-6 * jnp.eye(self.D)
      
    def get_dot_cov(self, t, weights_R, weights_Lambda, knots):

        cov_at_t = self.get_cov(self, t, weights_R, weights_Lambda, knots)
        return jax.jacobian(cov_at_t)(t)
    
    def fit_initial_covariance_weights(self, t, S_true, spline_basis, dim_D):
        """
        Stateless function to fit B-Spline weights for a Covariance model.
        
        Args:
            t: Time points (N,)
            S_true: Target covariance matrices (N, D, D)
            spline_basis: Initialized JAXBSpline object
            dim_D: Dimension of the covariance matrix
            
        Returns:
            W_L: Fitted log-variance weights
            W_R: Fitted rotation weights (skew-symmetric parameters)
        """
        phi = spline_basis.get_design_matrix(t) # (N, num_basis)
        
        # Decompose S_true: S = R @ Lambda @ R.T
        evals, evecs = jax.vmap(jnp.linalg.eigh)(S_true)
        
        # 1. Fit Log-Variances (Lambda)
        # Using clipping to ensure log is valid
        target_log_var = jnp.log(jnp.exp(jnp.maximum(evals, 1e-6)) - 1.0)
        w_l_fit, _, _, _ = jnp.linalg.lstsq(phi, target_log_var, rcond=None)
        W_L = w_l_fit.T  # (D, num_basis)

        # 2. Fit Rotation Parameters (R)
        # Since matrix log is complex, we extract rotation parameters by mapping
        # the eigenvectors directly to the skew-symmetric space.
        # For initialization, R_i is often approximated via the cross-product 
        # of the identity matrix and the eigenvector basis.
        i_idx, j_idx = jnp.triu_indices(dim_D, k=1)
        
        # Extracting rotation parameters: map eigenvectors to skew-symmetric components
        # We use evecs[:, i, j] as a projection of the rotation target
        target_R_vec = evecs[:, i_idx, j_idx] 
        
        w_r_fit, _, _, _ = jnp.linalg.lstsq(phi, target_R_vec, rcond=None)
        W_R = w_r_fit.T  # (dim_R, num_basis)
        
        return W_L, W_R