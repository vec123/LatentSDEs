import jax
import jax.numpy as jnp

class GaussianPathSDE:
    def __init__(self, mean_fn, dmean_dt_fn, Sigma_fn, dSigma_dt_fn, G_fn):
        """
        Computes the drift and diffusion for a Gaussian process path defined by mean_fn and Sigma_fn.
        mean_fn: function of time t that returns the mean vector (shape (D,))
        Sigma_fn: function of time t that returns the covariance matrix (shape (D, D))
        G_fn: function of time t that returns the diffusion matrix (shape (D, D))  
        """
        self.__module__
        self.D = mean_fn(0.0).shape[0]  # Assuming mean_fn returns a vector of shape (D,)
        self.mean_fn = mean_fn
        self.Sigma_fn = Sigma_fn
        self.A_fn, self.b_fn = self.get_drift_and_bias(mean_fn, dmean_dt_fn, Sigma_fn, dSigma_dt_fn, G_fn)
        self.G_fn = G_fn

    def sde_drift(self, t, mu):
        return self.A_fn(t) @ mu + self.b_fn(t)

    def sde_diffusion(self, t, sigma):
        A = self.A_fn(t)
        G = self.G_fn(t)
        # Lyapanov equation component: A*Sigma + Sigma*A^T + G*G^T
        return A @ sigma + sigma @ A.T + G @ G.T

    def get_drift_and_bias(self, mean_fn, dmean_dt_fn, Sigma_fn, dSigma_dt_fn, G_fn):
        """
        Computes both A(t) and b(t).
        
        Args:
            mean_fn: returns mu(t)
            dmean_dt_fn: returns dmu/dt(t)
            ... (previous args)
        """
        
        # 1. Compute A(t) using the spectral method as defined previously
        A_fn = self.get_drift_spectral(Sigma_fn, dSigma_dt_fn, G_fn)
        
        # 2. Compute b(t)
        def b_fn(t):
            mu = mean_fn(t)
            dmu_dt = dmean_dt_fn(t)
            A = A_fn(t)
            return dmu_dt - (A @ mu)
        
        return A_fn, b_fn

    def get_drift_spectral(self, Sigma_fn, dSigma_dt_fn, G_fn, ):
        """
        Implements the spectral parametrization drift calculation.
        
        Args:
            Sigma_fn: Callable returning Sigma(t) (D, D)
            dSigma_dt_fn: Callable returning dSigma/dt(t) (D, D)
            G_fn: Callable returning G(t) (D, D)
            R_op: Callable (vector of length D(D-1)/2) -> Skew-symmetric (D, D)
        """
        
        def A_fn(t):
            Sigma = Sigma_fn(t)
            dSigma_dt = dSigma_dt_fn(t)
            G = G_fn(t)
            
            # Q_theta = G * G.T
            Q = G @ G.T
            # Equation (16): solve (Lambda(t) (+) Lambda(t)) * vec(B) = vec(R.T * (Q - dSigma/dt) * R)
            # Note: Lambda(t) are eigenvalues of Sigma(t). 
            # Sigma = R * Lambda * R.T => Lambda = R.T * Sigma * R
            eigvals, eigvecs = jnp.linalg.eigh(Sigma)
            Lambda = jnp.diag(eigvals)
            R = eigvecs
            
            # RHS = R.T @ (Q - dSigma_dt) @ R
            rhs_matrix = R.T @ (Q - dSigma_dt) @ R
            
            # The system is diagonal in the spectral basis:
            # B_ij = RHS_ij / (lambda_i + lambda_j)
            # We add a small epsilon for numerical stability
            eps = 1e-8
            denom = eigvals[:, None] + eigvals[None, :]
            B = rhs_matrix / (denom + eps)
            
            # Reconstruct A = R * B * R.T
            A = R @ B @ R.T
            return A

        return A_fn
    
    def step(self, t, mu, sigma, dt):
        mu_new = mu + self.sde_drift(t, mu) * dt
        sigma_new = sigma + self.sde_sigma(t, sigma) * dt
        return mu_new, sigma_new
