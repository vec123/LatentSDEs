import jax
import jax.numpy as jnp
import haiku as hk

from src.sde.GaussianPaths.means.base import BaseMeanModel


class NNMeanModel(BaseMeanModel):
    def __init__(self, in_dim, out_dim, hidden_sizes, key=jax.random.PRNGKey(42)):
        self.in_dim = in_dim
        
        def fourier_features(t):
            t = jnp.atleast_1d(t) 
            
            freqs = jnp.linspace(0.1, 5.0, 10) 
            
            features = jnp.concatenate([
                jnp.sin(freqs[:, None] * t), 
                jnp.cos(freqs[:, None] * t)
            ], axis=0).flatten() # Result is (20,)
            
            return features

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
    
    def get_dot_mu(self, t):

        def forward(ti):
            return self.apply_fn(self.params, None, jnp.atleast_1d(ti))
        
        return jax.jacobian(forward)(t).squeeze()
    
    #def get_dot_mu(self, t):
    #    # Add this helper so save_fit_plot can call it
    #    return jax.grad(lambda ti: self.apply_fn(self.params, None, ti).sum())(jnp.atleast_1d(t))