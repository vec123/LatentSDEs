import jax
import jax.numpy as jnp

class JAXBSpline:
    def __init__(self, knots, degree):
        self.knots = jnp.array(knots)
        self.k = degree
        self.num_basis = len(knots) - degree - 1

    def basis(self, t, i, k):
        """Cox-de Boor recursion."""
        # Base case: degree 0
        if k == 0:
            return jnp.where((t >= self.knots[i]) & (t < self.knots[i+1]), 1.0, 0.0)
        
        # Recursive step
        denom1 = self.knots[i+k] - self.knots[i]
        term1 = jnp.where(denom1 != 0, ((t - self.knots[i]) / denom1) * self.basis(t, i, k-1), 0.0)
        
        denom2 = self.knots[i+k+1] - self.knots[i+1]
        term2 = jnp.where(denom2 != 0, ((self.knots[i+k+1] - t) / denom2) * self.basis(t, i+1, k-1), 0.0)
        
        return term1 + term2

    def get_design_matrix(self, t):
        # Evaluate all basis functions at all time points t
        return jnp.stack([self.basis(t, i, self.k) for i in range(self.num_basis)], axis=-1)