import jax
import jax.numpy as jnp

class GaussianPathSDE:
    def __init__(self, A_fn, b_fn, G_fn):
        """
        A(t), b(t), G(t) are callable functions (parameterized by NN, kernels, etc.)
        """
        self.A_fn = A_fn
        self.b_fn = b_fn
        self.G_fn = G_fn

    def drift_mu(self, t, mu):
        return self.A_fn(t) @ mu + self.b_fn(t)

    def drift_sigma(self, t, sigma):
        A = self.A_fn(t)
        G = self.G_fn(t)
        # Lyapanov equation component: A*Sigma + Sigma*A^T + G*G^T
        return A @ sigma + sigma @ A.T + G @ G.T

    def step(self, t, mu, sigma, dt):
        mu_new = mu + self.drift_mu(t, mu) * dt
        sigma_new = sigma + self.drift_sigma(t, sigma) * dt
        return mu_new, sigma_new

class SpectralGaussianSDE:
    def __init__(self, D, basis_fn):
        self.D = D
        self.basis_fn = basis_fn # e.g., Matern 5/2 feature vector k(t)

    def get_params(self, t, phi_R, phi_Lambda):
        # Basis expansion: phi(t) = k(t) @ weights
        r_vec = self.basis_fn(t) @ phi_R
        l_vec = self.basis_fn(t) @ phi_Lambda
        
        S_t = get_S_t(r_vec, l_vec, self.D)
        return S_t

    def get_A(self, t, S_t, S_dot, Q):
        return compute_drift_A(S_t, S_dot, Q)