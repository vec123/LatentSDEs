from src.sde.tests.data_helper import generate_data
from src.sde.tests.plot_helper import save_fit_plot
import jax.numpy as jnp
from src.sde.GaussianPaths.means.b_spline_mean import BSplineMeanModel

trajs, t_vals = generate_data()
t, y = trajs[0]
print("t.shape: ", t.shape)
print("y.shape: ", y.shape)
NUM_KNOTS = 100
knots = jnp.linspace(t.min(), t.max(), NUM_KNOTS)
model = BSplineMeanModel(t, knots, y=y, degree=3)
print(f"B-Spline Model initialized. Prediction at t=1.0: {model(1.0)}")
eps = 1e-6
t_eval = jnp.linspace(t.min() + eps, t.max() - eps, 500)
save_fit_plot(model, t, y, "b_spline_fit.png")