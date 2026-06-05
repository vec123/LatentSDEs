import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
from src.sde.tests.data_helper import generate_data
from src.sde.tests.plot_helper import plot_ensemble
from src.sde.GaussianPaths.means.nn_mean import NNMeanModel

def train_on_ensemble(model, trajs, epochs=100):
    # Flatten all trajectories into one dataset
    all_t = jnp.concatenate([t for t, y in trajs])
    all_y = jnp.concatenate([y for t, y in trajs])
    
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(model.params)
    
    def loss_fn(params):
        # Predict y for all points in all trajectories at once
        y_pred = jax.vmap(lambda ti: model.apply_fn(params, None, ti))(all_t[:, None])
        return jnp.mean((y_pred.flatten() - all_y)**2)

    for epoch in range(epochs):
        loss, grads = jax.value_and_grad(loss_fn)(model.params)
        model.params = optax.apply_updates(model.params, optimizer.update(grads, opt_state)[0])
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Ensemble Loss: {loss:.4f}")
    return model


# 1. Generate Ensemble
trajs, t_vals = generate_data(num_trajectories=10)

# 2. Train and Plot NN
model_nn = NNMeanModel( 1, 1, [32, 32])
model_nn = train_on_ensemble(model_nn, trajs) # Use the batch training function
plot_ensemble(model_nn, trajs, "NN Model Ensemble Fit", "nn_ensemble.png")

