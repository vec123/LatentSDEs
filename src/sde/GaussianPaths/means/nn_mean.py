import jax
import jax.numpy as jnp
import haiku as hk

from src.sde.GaussianPaths.means.base import BaseMeanModel

class NNMeanModel(BaseMeanModel):
    def __init__(self, time_points, Hm, in_dim, out_dim, hidden_sizes, key=jax.random.PRNGKey(42)):
        self.Hm = Hm
        self.in_dim = in_dim
        
        # INCREASED frequency range to cover 4*pi
        def fourier_features(t):
            # Using 10 frequencies that span the period of the data
            freqs = jnp.linspace(0.1, 2.0, 10) 
            return jnp.concatenate([jnp.sin(freqs * t), jnp.cos(freqs * t)], axis=-1)

        def model_fn(t):
            x = fourier_features(t)
            initializer = hk.initializers.VarianceScaling(2.0, mode='fan_in', distribution='truncated_normal')
            model = hk.Sequential([
                hk.Linear(hidden_sizes[0], w_init=initializer),
                jax.nn.silu,
                hk.Linear(hidden_sizes[1], w_init=initializer),
                jax.nn.silu,
                hk.Linear(out_dim, w_init=initializer)
            ])
            return model(x)
        
        self.transformed = hk.transform(model_fn)
        self.apply_fn = self.transformed.apply 
        
        # Initialize with dummy input
        dummy_input = jnp.zeros((1,)) 
        self.params = self.transformed.init(key, dummy_input)
        
    def __call__(self, t):
        return self.apply_fn(self.params, None, jnp.atleast_1d(t))
    
