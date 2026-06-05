from src.sde.tests.data_helper import generate_data
from src.sde.tests.plot_helper import save_fit_plot
from src.sde.GaussianPaths.means.matern_52_mean import MaternMeanModel

trajs, t_vals = generate_data()
t, y = trajs[0]

# Fit
model = MaternMeanModel(t, y, length_scale=0.5, sigma_f=1.0)
print(f"Matern Model initialized. Prediction at t=1.0: {model(1.0)}")
save_fit_plot(model, t, y, "matern_fit.png")