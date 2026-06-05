import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from src.sde.GaussianPaths.means.matern_52_mean import MaternMeanModel
from src.sde.tests.data_helper import generate_data
from src.sde.tests.plot_helper import plot_ensemble

# 1. Generate Ensemble
trajs, t_vals = generate_data(num_trajectories=10)
all_t = jnp.concatenate([t for t, y in trajs])
all_y = jnp.concatenate([y for t, y in trajs])

# 2. Initialize Matern Model
# We fit this to all points in the ensemble simultaneously
model_matern = MaternMeanModel(all_t, all_y, length_scale=1, sigma_f=1.0)

# 3. Plotting the result
# The Matern model handles the mean and covariance of the ensemble
plot_ensemble(model_matern, trajs, "Matern Ensemble Fit", "matern_ensemble.png")