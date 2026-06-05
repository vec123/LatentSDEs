import jax
import jax.numpy as jnp
from src.models.b_splines import JAXBSpline 
from src.sde.GaussianPaths.means.b_spline_mean import BSplineMeanModel
from src.sde.tests.mean_field_plots import plot_model_path
from src.sde.tests.gp_ground_truths import plot_gp_ground_truth

def fit_and_plot():
    T = 11.0
    num_knots = 10
    degree = 3
    t_points = jnp.linspace(0, T, 40)
    eps = 1e-6
    t_eval =  jnp.linspace(eps, T-eps, 40)
    # Create sine wave oscillation around the linear diagonal
    freq = 2.0 * jnp.pi / 20
    dim1 = jnp.sin(freq * t_eval) * 0.5
    dim2 = jnp.cos(freq * t_eval) * 0.5

    # 3. Combine into (N, 2)
    # Stack them side-by-side
    target_y = jnp.stack([dim1, dim2], axis=1)
    
    # 3. Initialize model with target_y
    # Passing 'y=target_y' triggers the internal least-squares fit
    print("t_points.shape: ", t_points.shape)
    print("target_y.shape: ", target_y.shape)
    model = BSplineMeanModel(t=t_points, y=target_y, num_knots=num_knots, degree=degree)

    # Visualize the initialized path
    plot_model_path(t_eval, model, y_truth = target_y, file_name="b_spline_model_path_init.png")
if __name__ == "__main__":
    fit_and_plot()