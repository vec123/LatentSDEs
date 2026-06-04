from src.sde.tests.data_helper import generate_data
from src.sde.tests.plot_helper import save_fit_plot
import jax.numpy as jnp
from src.sde.GaussianPaths.means.b_spline_mean import BSplineMeanModel
trajs = generate_data()
t, y = trajs[0]

model = BSplineMeanModel(t, y, num_knots=10, degree=3)
print(f"B-Spline Model initialized. Prediction at t=1.0: {model(1.0)}")
save_fit_plot(model, t, y, "b_spline_fit.png")