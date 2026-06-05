
import jax
import jax.numpy as jnp
from src.models.b_splines import JAXBSpline

class BSplineMeanModel_:
    def __init__(self, t, knots, y=None, v=None, a=None, coeffs=None, degree=3):
        t = jnp.asarray(t).flatten()
        knots = jnp.sort(knots)
        
        # Clamped knot vector
        self.knots_padded = jnp.concatenate([
            jnp.repeat(t.min(), degree + 1), 
            knots[1:-1], 
            jnp.repeat(t.max(), degree + 1)
        ])
        
        self.spline_basis = JAXBSpline(self.knots_padded, degree)
        self.deriv_basis = JAXBSpline(self.knots_padded[1:-1], degree - 1)
        self.dderiv_basis = JAXBSpline(self.knots_padded[2:-2], degree - 2)
        
        # --- Multi-Objective Least Squares ---
        A_list = []
        b_list = []
        
        if y is not None:
            A_list.append(self.spline_basis.get_design_matrix(t))
            b_list.append(jnp.asarray(y))
        if v is not None:
            A_list.append(self.deriv_basis.get_design_matrix(t))
            b_list.append(jnp.asarray(v))
        if a is not None:
            A_list.append(self.dderiv_basis.get_design_matrix(t))
            b_list.append(jnp.asarray(a))
            
        if coeffs is not None:
            self.coeffs = jnp.asarray(coeffs)
        elif A_list:
            A = jnp.concatenate(A_list, axis=0)
            b = jnp.concatenate(b_list, axis=0)
            self.coeffs, _, _, _ = jnp.linalg.lstsq(A, b, rcond=None)
        else:
            raise ValueError("Must provide y, v, a, or coeffs for initialization.")
        
        # --- Pre-compute Derivative Coefficients ---
        # First Derivative
        k = degree
        deriv_coeffs = []
        for i in range(len(self.coeffs) - 1):
            denom = self.knots_padded[i + k + 1] - self.knots_padded[i + 1]
            val = k * (self.coeffs[i + 1] - self.coeffs[i]) / (denom + 1e-10)
            deriv_coeffs.append(val)
        self.deriv_coeffs = jnp.array(deriv_coeffs)

        # Second Derivative
        dderiv_coeffs = []
        for i in range(len(self.deriv_coeffs) - 1):
            denom = self.knots_padded[i + k] - self.knots_padded[i + 1]
            val = (k - 1) * (self.deriv_coeffs[i + 1] - self.deriv_coeffs[i]) / (denom + 1e-10)
            dderiv_coeffs.append(val)
        self.dderiv_coeffs = jnp.array(dderiv_coeffs)

    def __call__(self, t):
        t = jnp.atleast_1d(t)
        phi_t = self.spline_basis.get_design_matrix(t)
        return jnp.dot(phi_t, self.coeffs).reshape(-1)

    def get_dot_mu(self, t):
        phi_dot = self.deriv_basis.get_design_matrix(jnp.atleast_1d(t))
        return jnp.dot(phi_dot, self.deriv_coeffs).reshape(-1)
    
    def get_ddot_mu(self, t):
        phi_ddot = self.dderiv_basis.get_design_matrix(jnp.atleast_1d(t))
        return jnp.dot(phi_ddot, self.dderiv_coeffs).reshape(-1)



class BSplineMeanModel:
    def __init__(self, t, knots, y=None, coeffs = None, degree=3):
        t = jnp.asarray(t).flatten()
        self.knots = jnp.sort(knots)
        # Use a clamped knot vector where knots at start/end are repeated
        # AND ensure the last knot is exactly t.max()
        self.knots_padded = jnp.concatenate([
            jnp.repeat(t.min(), degree + 1), 
            knots[1:-1], # Interior knots
            jnp.repeat(t.max(), degree + 1)
        ])
        self.spline_basis = JAXBSpline(self.knots_padded, degree)
        self.deriv_basis = JAXBSpline(self.knots_padded[1:-1], degree - 1)
        self.dderiv_basis = JAXBSpline(self.knots_padded[2:-2], degree - 2)
        
        # Least Squares fit
        phi = self.spline_basis.get_design_matrix(t)
        if y is not None:
            # y is (T, D). DO NOT flatten.
            y = jnp.asarray(y) 
            # lstsq solves (T, M) @ (M, D) = (T, D)
            # self.coeffs will be (M, D)
            self.coeffs, _, _, _ = jnp.linalg.lstsq(phi, y, rcond=None)

        if coeffs is not None:
            self.coeffs = jnp.asarray(coeffs)
        if coeffs is not None and y is not None:
            raise ValueError("Provide either coeffs or y, not both.")
        if self.coeffs is None:
            raise ValueError("Must provide either coeffs directly or y for initialization via least squares fit.")
        
        # Pre-compute derivative basis once!
        k = degree
        new_coeffs = []
        for i in range(len(self.coeffs) - 1):
            denom = self.knots_padded[i + k + 1] - self.knots_padded[i + 1]
            val = k * (self.coeffs[i + 1] - self.coeffs[i]) / (denom + 1e-10)
            new_coeffs.append(val)
        self.deriv_coeffs = jnp.array(new_coeffs)

         # Pre-compute double derivative basis once!
        dderiv_coeffs = []
        for i in range(len(self.deriv_coeffs) - 1):
            denom = self.knots_padded[i + k] - self.knots_padded[i + 1]
            val = (k - 1) * (self.deriv_coeffs[i + 1] - self.deriv_coeffs[i]) / (denom + 1e-10)
            dderiv_coeffs.append(val)
        self.dderiv_coeffs = jnp.array(dderiv_coeffs)
        self.dderiv_basis = JAXBSpline(self.knots_padded[2:-2], k - 2)


    def __call__(self, t):
        # Ensure t is a scalar for the gradient to work
        t = jnp.atleast_1d(t)
        phi_t = self.spline_basis.get_design_matrix(t)
        # Result is shape (1,) or (N,), return scalar if input was scalar
        res = jnp.dot(phi_t, self.coeffs)
        return res.reshape(-1)

    def get_dot_mu(self, t):
        phi_dot = self.deriv_basis.get_design_matrix(jnp.atleast_1d(t))
        return jnp.dot(phi_dot, self.deriv_coeffs).reshape(-1)
    
    def get_ddot_mu(self, t):
        phi_dot = self.dderiv_basis.get_design_matrix(jnp.atleast_1d(t))
        return jnp.dot(phi_dot, self.dderiv_coeffs).reshape(-1)
    