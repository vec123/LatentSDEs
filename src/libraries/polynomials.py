import jax.numpy as jnp

class BasisLibrary:
    """
    A library to generate basis function matrices for regression models.
    """
    
    @staticmethod
    def polynomial(x, degree=3):
        """
        Generates a polynomial feature matrix.
        Input x: shape (n,)
        Output: shape (n, degree)
        """
        # Ensure x is a column vector
        x = x.reshape(-1, 1)
        
        # Create powers: [x^1, x^2, ..., x^degree]
        # We start from 1 to avoid the intercept (often handled separately)
        powers = jnp.arange(1, degree + 1)
        return jnp.power(x, powers)