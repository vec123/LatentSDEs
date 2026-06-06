import jax
import jax.numpy as jnp
import optax
from jax.nn import softplus
from src.distributions.learnable import init_gaussian_params
from src.distributions.horsehoe import basic_horseshoe

# --- 1. Learnable Gaussian ---
def init_gaussian_params(key, dim):
    return {'mu': jax.random.normal(key, (dim,)), 
            'log_sigma': jnp.zeros(dim)}

def get_gaussian_entropy(log_sigma):
    return 0.5 * jnp.sum(jnp.log(2 * jnp.pi * jnp.exp(2 * log_sigma)) + 1)

# --- 2. KL Divergence Terms (Derived) ---
def kl_ig(mu, log_sigma, a, b):
    # D_KL(q(x)||IG(a,b)) for log-normal q(x)
    var = jnp.exp(2 * log_sigma)
    return -0.5 * (jnp.log(2 * jnp.pi * var) + 1) - (a * jnp.log(b) - jax.scipy.special.gammaln(a) - (a + 1) * mu - b * jnp.exp(-mu + var / 2))

def kl_gamma(mu, log_sigma, a, b):
    # D_KL(q(x)||G(a,b)) for log-normal q(x)
    var = jnp.exp(2 * log_sigma)
    return -0.5 * (jnp.log(2 * jnp.pi * var) + 1) - (a * jnp.log(b) - jax.scipy.special.gammaln(a) + (a - 1) * mu - b * jnp.exp(mu + var / 2))

# --- 3. Full Horseshoe Loss ---
def horseshoe_loss(params, D=10):
    # theta ~ N(0, z^2 * s^2)
    # s = sa * sb; z = za * zb
    s_a = params['s_a']; s_b = params['s_b']
    z_a = params['z_a']; z_b = params['z_b']
    
    # KL for theta_i
    z_sq_s_sq = jnp.exp(2*(z_a['mu'] + z_b['mu'] + s_a['mu'] + s_b['mu']))
    kl_theta = 0.5 * jnp.sum(jnp.log(z_sq_s_sq) - params['theta']['log_sigma'] - 1 + 
                             (jnp.exp(2*params['theta']['log_sigma']) + params['theta']['mu']**2) / (z_sq_s_sq + 1e-8))
    
    # KL for scales
    kl_scales = kl_gamma(s_a['mu'], s_a['log_sigma'], 0.5, 1.0) + \
                kl_gamma(s_b['mu'], s_b['log_sigma'], 0.5, 1.0) + \
                jnp.sum(kl_gamma(z_a['mu'], z_a['log_sigma'], 0.5, 1.0)) + \
                jnp.sum(kl_gamma(z_b['mu'], z_b['log_sigma'], 0.5, 1.0))
                
    return kl_theta + kl_scales

# --- 4. Main Training Loop ---
def fit_horseshoe():
    D = 10
    key = jax.random.PRNGKey(42)
    
    params = {
        'theta': init_gaussian_params(key, D),
        's_a': init_gaussian_params(key, 1),
        's_b': init_gaussian_params(key, 1),
        'z_a': init_gaussian_params(key, D),
        'z_b': init_gaussian_params(key, D)
    }
    
    optimizer = optax.adam(0.01)
    opt_state = optimizer.init(params)
    
    @jax.jit
    def step(params, opt_state):
        loss_val, grads = jax.value_and_grad(horseshoe_loss)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss_val

    for i in range(2000):
        params, opt_state, loss = step(params, opt_state)
        if i % 200 == 0:
            print(f"Step {i}, Loss: {loss:.4f}")

if __name__ == "__main__":
    fit_horseshoe()