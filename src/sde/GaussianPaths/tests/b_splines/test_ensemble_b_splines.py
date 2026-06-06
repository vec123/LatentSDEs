import jax.numpy as jnp
from src.sde.tests.data_helper import generate_data
from src.sde.tests.plot_helper import plot_ensemble
from src.sde.GaussianPaths.means.b_spline_mean import BSplineMeanModel

NUM_KNOTS =15
# 1. Generate Ensemble
trajs, t_vals = generate_data(num_trajectories=10)
all_t = jnp.concatenate([t for t, y in trajs])
all_y = jnp.concatenate([y for t, y in trajs])

# 2. Fit B-Spline to Ensemble
# We use the JAX-native implementation created previously
knots = jnp.linspace(t_vals.min(), t_vals.max(), NUM_KNOTS)
model_bspline = BSplineMeanModel(all_t, knots, all_y, degree=3)

# 3. Plotting
plot_ensemble(model_bspline, trajs, "B-Spline Ensemble Fit", "bspline_ensemble.png")
print("B-Spline Ensemble training complete.")