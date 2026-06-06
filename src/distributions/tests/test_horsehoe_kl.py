import jax
import jax.numpy as jnp
import optax
from jax.nn import softplus

from src.distributions.learnable import init_gaussian_params
from src.distributions.divergences import kl_ig, kl_gamma

# --- Core Model: Horseshoe Posterior ---
class HorseshoePosterior:
    def __init__(self, D, tau_0 = 1,  
                 local_shape=2.0, local_rate=1.0,
                 global_shape=0.5, global_rate=1.0):
        """
        A Variational Horseshoe Posterior model for sparse feature selection.
        
        Hyperparameters for tuning signal sensitivity:
        - tau_0: Initial guess for the global scale. Higher values permit larger signals.
        - local_shape: Controls the 'spikiness' of the local prior. 
                    Set to 0.5 for extreme sparsity; increase to >2.0 to retain smaller signals.
        - local_rate: Scales the variance of the local distribution. 
                    Increasing this spreads the prior mass, allowing more parameters to drift from zero.
        - global_shape: Controls the global shrinkage strength.
        - global_rate: Adjusts the global shrinkage scale.
        """
        
        self.D = D
        self.tau_0 = tau_0
        self.local_shape = local_shape
        self.local_rate = local_rate
        self.global_shape = global_shape
        self.global_rate = global_rate

    def init_params(self, key):
            k1, k2, k3, k4, k5 = jax.random.split(key, 5)
            # Pass type='diagonal' to ensure 'log_var' is created
            return {
                'theta': init_gaussian_params(k1, self.D, type='diagonal'),
                's_a': init_gaussian_params(k2, 1, type='diagonal'),
                's_b': init_gaussian_params(k3, 1, type='diagonal'),
                'z_a': init_gaussian_params(k4, self.D, type='diagonal'),
                'z_b': init_gaussian_params(k5, self.D, type='diagonal')
            }
    
    def kl_divergence(self, params):
        # Helper to get the correct log_sigma from whatever init_params returned
        def get_log_sigma(p):
            if 'log_sigma' in p: return p['log_sigma']
            if 'log_var' in p: return 0.5 * p['log_var']
            raise KeyError("Parameters must contain either 'log_sigma' or 'log_var'")

        rate = 1.0 / (self.tau_0**2)
        # Extract parameters
        s_a, s_b, z_a, z_b, theta = params['s_a'], params['s_b'], params['z_a'], params['z_b'], params['theta']
        
        # Calculate log_sigma for all components
        ls_theta = get_log_sigma(theta)
        ls_s_a = get_log_sigma(s_a)
        ls_s_b = get_log_sigma(s_b)
        ls_z_a = get_log_sigma(z_a)
        ls_z_b = get_log_sigma(z_b)
        
        # Use these in your KL formulas
        log_z_s = z_a['mu'] + z_b['mu'] + s_a['mu'] + s_b['mu']
        z_sq_s_sq = jnp.exp(2 * log_z_s)
        
        kl_theta = 0.5 * jnp.sum(
            (2 * log_z_s) - ls_theta - 1 + 
            (jnp.exp(2 * ls_theta) + theta['mu']**2) / (z_sq_s_sq + 1e-8)
        )
        
        kl_scales = (
            kl_gamma(s_a['mu'], ls_s_a, self.global_shape, rate) +
            kl_ig(s_b['mu'], ls_s_b, self.global_shape, self.global_rate) +
            jnp.sum(kl_gamma(z_a['mu'], ls_z_a, self.local_shape, self.local_rate)) +
            jnp.sum(kl_ig(z_b['mu'], ls_z_b, self.local_shape, self.local_rate))
        )
        return kl_theta + kl_scales

    def loss(self, params, y):
            # ELBO = NLL + KL
            nll = jnp.sum((params['theta']['mu'] - y)**2) 
            total_loss = nll + self.kl_divergence(params)
            
            # Ensure the output is a 0-D scalar
            return jnp.sum(total_loss)
    
    def sample_prior(self, key):
        """Draws a sample from the Horseshoe prior distribution."""
        k1, k2, k3, k4 = jax.random.split(key, 4)
        
        # Draw scales from Gamma and Inverse-Gamma priors
        s_a = jax.random.gamma(k1, 0.5) / 1.0  # Gamma(0.5, 1.0)
        s_b = 1.0 / jax.random.gamma(k2, 0.5)  # IG(0.5, 1.0) -> 1/Gamma
        z_a = jax.random.gamma(k3, 0.5, shape=(self.D,))
        z_b = 1.0 / jax.random.gamma(k4, 0.5, shape=(self.D,))
        
        # Combine scales: theta ~ N(0, (z_a*z_b)^2 * (s_a*s_b)^2)
        scale = (z_a * z_b) * (s_a * s_b)
        theta = jax.random.normal(k1, (self.D,)) * scale
        return theta
    
    def sample_posterior(self, key, params):
        """Draws a sample from the variational posterior q(theta)."""
        # theta ~ N(mu, sigma^2)
        mu = params['theta']['mu']
        sigma = jnp.exp(params['theta']['log_sigma'])
        noise = jax.random.normal(key, shape=(self.D,))
        return mu + sigma * noise
    
    def get_uncertainty(self, params):
        """Returns the mean and standard deviation for the posterior."""
        mu = params['theta']['mu']
        # Convert log_var to standard deviation: std = exp(0.5 * log_var)
        std = jnp.exp(0.5 * params['theta']['log_var'])
        return mu, std
# --- Execution ---
def fit_horseshoe():
    D = 10
    model = HorseshoePosterior(D, tau_0=1000000)
    key = jax.random.PRNGKey(42)
    
    # Dummy data: Sparse signal
    y = jnp.array([6.0, 0.0, 0.5, 0.0, 0.0, 12.0, 0.0, 0.0, 0.0, 0.0])
    y = y / jnp.std(y)
    params = model.init_params(key)
    optimizer = optax.adam(0.01)
    opt_state = optimizer.init(params)
    
    @jax.jit
    def step(params, opt_state):
        loss_val, grads = jax.value_and_grad(model.loss)(params, y)
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss_val

    for i in range(4001):
        params, opt_state, loss = step(params, opt_state)
        if i % 200 == 0:
            print(f"Step {i}, Loss: {loss:.4f}, Global Scale Mean: {params['s_a']['mu'][0]:.4f}")
            print(f"Step {i}, Loss: {loss:.4f}, Theta Means: {params['theta']['mu']}")

    mu, std = model.get_uncertainty(params)
    print("\n--- Final Uncertainty Check ---")
    for i in range(D):
        print(f"Index {i}: Mean = {mu[i]:.4f}, Std Dev = {std[i]:.4f}")

if __name__ == "__main__":
    fit_horseshoe()