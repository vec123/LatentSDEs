import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_sylvester

def skew_symmetric(vec, D):
    """Turns a vector of length D(D-1)/2 into a skew-symmetric matrix."""
    mat = jnp.zeros((D, D))
    # Indices for the upper triangular part
    idx = 0
    for i in range(D):
        for j in range(i + 1, D):
            val = vec[idx]
            mat = mat.at[i, j].set(val)
            mat = mat.at[j, i].set(-val)
            idx += 1
    return mat

def get_S_t(R_vec, Lambda_vec, D):
    """
    Parametrizes S(t) = R(t) * Lambda(t) * R(t)^T
    R(t) = exp(skew(R_vec))
    Lambda(t) = diag(softplus(Lambda_vec))
    """
    R = jax.scipy.linalg.expm(skew_symmetric(R_vec, D))
    Lambda = jnp.diag(jax.nn.softplus(Lambda_vec))
    return R @ Lambda @ R.T

def compute_drift_A(S_t, S_dot, Q):
    """
    Solves A * S + S * A^T = S_dot - Q for A.
    Uses Bartels-Stewart / Sylvester solver.
    """
    # Sylvester: A*S + S*A^T = (S_dot - Q)
    # This matches the form AX + XB = C
    # Here X=A, B=S, C=(S_dot - Q) -> A*S + S*A^T = S_dot - Q
    # Note: jax.scipy.linalg.solve_sylvester solves AX + XB = C
    # To map to A*S + S*A^T = (S_dot - Q):
    # A*S + S*A^T = RHS => A*S + S.T*A^T = RHS
    return solve_sylvester(jnp.eye(S_t.shape[0]), S_t, (S_dot - Q))