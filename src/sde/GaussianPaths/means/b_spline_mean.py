
import jax
import jax.numpy as jnp
from src.models.b_splines import JAXBSpline

class BSplineMeanModel:
    def __init__(self, t, knots, y=None, coeffs = None, degree=3):

        t = jnp.asarray(t).flatten()
        self.knots = jnp.sort(knots)
        self.knots_padded = jnp.concatenate([
            jnp.repeat(t.min(), degree + 1), 
            knots[1:-1],
            jnp.repeat(t.max(), degree + 1)
        ])
        self.spline_basis = JAXBSpline(self.knots_padded, degree)
        self.deriv_basis = JAXBSpline(self.knots_padded[1:-1], degree - 1)
        self.dderiv_basis = JAXBSpline(self.knots_padded[2:-2], degree - 2)
        
        #If Least Squares fit
        phi = self.spline_basis.get_design_matrix(t)
        if y is not None:
            y = jnp.asarray(y) 
            self.coeffs, _, _, _ = jnp.linalg.lstsq(phi, y, rcond=None)

        if coeffs is not None:
            self.coeffs = jnp.asarray(coeffs)
        if coeffs is not None and y is not None:
            raise ValueError("Provide either coeffs or y, not both.")
        if self.coeffs is None:
            raise ValueError("Must provide either coeffs directly or y for initialization via least squares fit.")
        
        # Pre-compute derivative basis once!
        self.deriv_coeffs, self.knots_deriv = self.compute_spline_derivative_coeffs(
            self.coeffs, self.knots_padded, degree
        )
        self.deriv_basis = JAXBSpline(self.knots_deriv, degree - 1)

        # Second Derivative (Recursion)
        self.dderiv_coeffs, self.knots_dderiv = self.compute_spline_derivative_coeffs(
            self.deriv_coeffs, self.knots_deriv, degree - 1
        )
        self.dderiv_basis = JAXBSpline(self.knots_dderiv, degree - 2)
        
    def compute_spline_derivative_coeffs(self, coeffs, knots_padded, degree):
        k = degree
        deriv_coeffs = []
        
        # Derivative formula for B-Splines:
        # d/dt B_{i,k}(t) = k * (B_{i+1, k-1}(t) - B_{i, k-1}(t)) / (knots[i+k+1] - knots[i+1])
        for i in range(len(coeffs) - 1):
            denom = knots_padded[i + k + 1] - knots_padded[i + 1]
            val = k * (coeffs[i + 1] - coeffs[i]) / (denom + 1e-10)
            deriv_coeffs.append(val)
            
        # The derivative of a B-Spline of degree k is a B-Spline of degree k-1.
        # The knot vector for the derivative drops the first and last padding knots.
        new_knots = knots_padded[1:-1]
        return jnp.array(deriv_coeffs), new_knots
    
    def __call__(self, t):
        t = jnp.atleast_1d(t)
        phi_t = self.spline_basis.get_design_matrix(t)
        res = jnp.dot(phi_t, self.coeffs)
        return res.reshape(-1)

    def get_dot_mu(self, t):
        phi_dot = self.deriv_basis.get_design_matrix(jnp.atleast_1d(t))
        return jnp.dot(phi_dot, self.deriv_coeffs).reshape(-1)
    
    def get_ddot_mu(self, t):
        phi_dot = self.dderiv_basis.get_design_matrix(jnp.atleast_1d(t))
        return jnp.dot(phi_dot, self.dderiv_coeffs).reshape(-1)
    