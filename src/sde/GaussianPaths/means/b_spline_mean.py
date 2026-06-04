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

class BSplineMeanModel:
    def __init__(self, t, y, num_knots=5, degree=3):
        t = jnp.asarray(t).flatten()
        y = jnp.asarray(y).flatten()
        
        # Setup knots
        knots = jnp.linspace(t.min(), t.max(), num_knots)
        self.knots_padded = jnp.concatenate([
            jnp.repeat(t.min(), degree), knots, jnp.repeat(t.max(), degree)
        ])
        
        self.spline_basis = JAXBSpline(self.knots_padded, degree)
        
        # Least Squares fit
        phi = self.spline_basis.get_design_matrix(t)
        self.coeffs, _, _, _ = jnp.linalg.lstsq(phi, y, rcond=None)

    def __call__(self, t):
        # Ensure t is a scalar for the gradient to work
        t = jnp.atleast_1d(t)
        phi_t = self.spline_basis.get_design_matrix(t)
        # Result is shape (1,) or (N,), return scalar if input was scalar
        res = jnp.dot(phi_t, self.coeffs)
        return res.squeeze()

    def get_dot_mu(self, t):
            # 1. Get current degree and knots
            k = self.spline_basis.k
            knots = self.spline_basis.knots
            
            # 2. Calculate derivative coefficients (degree k-1)
            # c'_i = k * (c_{i+1} - c_i) / (t_{i+k+1} - t_{i+1})
            # Note: We need to ensure knot indices align for degree k-1
            coeffs = self.coeffs
            new_coeffs = []
            for i in range(len(coeffs) - 1):
                denom = knots[i + k + 1] - knots[i + 1]
                # Use a small constant to prevent division by zero
                val = k * (coeffs[i + 1] - coeffs[i]) / (denom + 1e-10)
                new_coeffs.append(val)
            
            # 3. Evaluate the derivative using a B-spline of degree k-1
            # Create a temporary basis evaluator for degree k-1
            deriv_basis = JAXBSpline(knots[1:-1], k - 1)
            phi_dot = deriv_basis.get_design_matrix(t)
            
            return jnp.dot(phi_dot, jnp.array(new_coeffs))