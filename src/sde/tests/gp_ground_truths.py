import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

def get_ground_truth_gp(t):
    # Mean
    mu = jnp.array([jnp.sin(t), jnp.cos(t)])
    
    # Covariance: rotating ellipse
    theta = t * 0.5
    R = jnp.array([[jnp.cos(theta), -jnp.sin(theta)], 
                   [jnp.sin(theta),  jnp.cos(theta)]])
    Lambda = 10*jnp.diag(jnp.array([0.1 + 0.05 * jnp.sin(t), 0.1 + 0.05 * jnp.sin(t)]))
    S = R @ Lambda @ R.T
    S = 0.4*jnp.diag(jnp.array([1.1 + jnp.sin(t), 1.1 +  jnp.sin(t)]))
    return mu, S


def sample_trajectories(t_points, num_trajs=50):
    # 1. Create a JAX PRNG key
    key = jax.random.PRNGKey(42)
    trajs = []
    
    for _ in range(num_trajs):
        # 2. Split the key for every sample/trajectory to maintain JAX purity
        key, subkey = jax.random.split(key)
        y = []
        for t in t_points:
            mu, S = get_ground_truth_gp(t)
            # 3. Use jax.random.multivariate_normal
            # Note: multivariate_normal needs a key and handles the sampling
            sample = jax.random.multivariate_normal(subkey, mu, S)
            y.append(sample)
        trajs.append((t_points, jnp.array(y)))
    return trajs


def get_ground_truth_cov(t, mode = "full"):
    
    if mode == "full":
        # Define time-varying eigenvalues (must be positive)
        l1 = 1.0 + 0.5 * jnp.sin(t)
        l2 = 0.5 + 0.2 * jnp.cos(t)  
        # Define a rotation matrix based on t
        theta = 0.5 * t
        R = jnp.array([
                [jnp.cos(theta), -jnp.sin(theta)],
                [jnp.sin(theta),  jnp.cos(theta)]
        ])
            
        #  Construct covariance as R * diag(l1, l2) * R^T
        # This guarantees symmetry and positive definiteness
        diag_l = jnp.diag(jnp.array([l1, l2]))
        S = R @ diag_l @ R.T    
    elif mode =="diagonal":
        S = 0.4*jnp.diag(jnp.array([1.1 + jnp.sin(t), 1.1 +  jnp.sin(t)]))
    else:
        raise ValueError("mode must be full or diagonal")

    return S

