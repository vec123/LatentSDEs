import jax
import jax.numpy as jnp
import haiku as hk
from jax.scipy.linalg import expm

class NNCovariance(hk.Module):
    def __init__(self, D, hidden_sizes=(64, 64), type='full'):
        super().__init__()
        self.D = D
        self.type = type
        # The NN outputs: 
        # 1. Log-diagonal elements (D)
        # 2. Skew-symmetric weights (D*(D-1)/2) if 'full'
        out_dim = D + (D*(D-1)//2 if type == 'full' else 0)
        self.model = hk.Sequential([
            hk.Linear(hidden_sizes[0]), jax.nn.silu,
            hk.Linear(hidden_sizes[1]), jax.nn.silu,
            hk.Linear(out_dim)
        ])

    def _to_skew_symmetric(self, vec):
        mat = jnp.zeros((self.D, self.D))
        i_idx, j_idx = jnp.triu_indices(self.D, k=1)
        mat = mat.at[i_idx, j_idx].set(vec)
        mat = mat.at[j_idx, i_idx].set(-vec)
        return mat

    def __call__(self, t):
        out = self.model(jnp.atleast_1d(t))
        
        # Split output into diagonal and rotation params
        log_diag = out[:self.D]
        Lambda = jnp.diag(jnp.exp(log_diag))
        
        if self.type == 'diagonal':
            return Lambda
        
        # Construct full covariance
        vec_R = out[self.D:]
        omega = self._to_skew_symmetric(vec_R)
        R = expm(omega)
        return R @ Lambda @ R.T