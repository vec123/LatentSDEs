import jax
import jax.numpy as jnp

class SDE:
    def __init__(self, drift_fn, diffusion_fn):
        """
        Generic SDE class.
        :param drift_fn: callable f(t, x) -> drift
        :param diffusion_fn: callable g(t, x) -> diffusion
        """
        self.f = drift_fn
        self.g = diffusion_fn

    def step(self, t, x, dt, key):
        """
        Perform a single Euler-Maruyama integration step.
        """
        drift = self.f(t, x)
        diffusion = self.g(t, x)
        
        # Weiner increment dW ~ N(0, dt)
        dw = jax.random.normal(key, shape=x.shape) * jnp.sqrt(dt)
        
        # X_{t+dt} = X_t + f(t, X_t)dt + g(t, X_t)dW_t
        x_next = x + drift * dt + diffusion * dw
        return x_next

    def simulate(self, x0, t_span, dt, key):
        """
        Simulate a trajectory over a time span.
        """
        steps = int((t_span[1] - t_span[0]) / dt)
        
        def body_fn(carry, k):
            t, x = carry
            x_next = self.step(t, x, dt, k)
            return (t + dt, x_next), x_next

        keys = jax.random.split(key, steps)
        _, trajectory = jax.lax.scan(body_fn, (t_span[0], x0), keys)
        return trajectory