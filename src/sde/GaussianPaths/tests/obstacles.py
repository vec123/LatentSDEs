import jax.numpy as jnp

def get_obstacle_loss(x_t, obstacles):
    total_pot = 0.0
    for obs in obstacles:
        center = jnp.array(obs['center'])
        dist = jnp.linalg.norm(x_t - center)
        
        #  Define a buffer zone (radius + small margin)
        margin = obs['radius'] + 3
        
        # Use a quadratic hinge:
        # If dist > margin, potential is 0.
        # If dist < margin, potential grows quadratically as we enter the buffer.
        # This is much smoother than exp() and prevents gradient explosion.
        diff = margin - dist
        total_pot += jnp.where(dist < margin, 0.5 * (jnp.abs(diff)**2), 0.0)
    return 5*total_pot