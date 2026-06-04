import jax
import jax.numpy as jnp
import optax
from src.sde.tests.plot_helper import save_fit_plot
from src.sde.tests.data_helper import generate_data
from src.sde.GaussianPaths.means.nn_mean import NNMeanModel

def train_nn(model, t, y, epochs=500):
    optimizer = optax.adam(1e-3) # Lower learning rate for stability
    opt_state = optimizer.init(model.params)
    
    # Shuffle indices to remove temporal bias and ensure equal weighting
    key = jax.random.PRNGKey(0)
    indices = jax.random.permutation(key, len(t))
    t_shuffled, y_shuffled = t[indices][:, None], y[indices]
    
    def loss_fn(params):
        # vmap over shuffled batches
        y_pred = jax.vmap(lambda ti: model.apply_fn(params, None, ti))(t_shuffled)
        return jnp.mean((y_pred.flatten() - y_shuffled)**2)

    for epoch in range(epochs):
        loss, grads = jax.value_and_grad(loss_fn)(model.params)
        model.params = optax.apply_updates(model.params, optimizer.update(grads, opt_state)[0])
        if epoch % 200 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    return model

trajs = generate_data()
t, y = trajs[0]
model = NNMeanModel(t, y, in_dim=1, out_dim=1, hidden_sizes=[32, 32])
model = train_nn(model, t, y)
print("NN Training complete.")
save_fit_plot(model, t, y, "nn_fit.png")