
import haiku as hk
import jax.numpy as jnp
import jax

def nn_forward(t, in_dim, out_dim, hidden_sizes):

    assert hidden_sizes[-1] ==  out_dim, \
        f"NN: Last hidden_size must be {out_dim}, got {hidden_sizes[-1]}"
    assert hidden_sizes[0] ==  in_dim, \
        f"NN: First hidden_size must be {in_dim}, got {hidden_sizes[0]}"

    layers = []
    # Input(1) -> first hidden layer
    layers.append(hk.Linear(hidden_sizes[0]))
    layers.append(jax.nn.relu)
    
    # Remaining hidden layers
    for i in range(len(hidden_sizes) - 1):
        layers.append(hk.Linear(hidden_sizes[i+1]))
        if i < len(hidden_sizes) - 2: # Keep activations between hidden layers
            layers.append(jax.nn.relu)
            
    mlp = hk.Sequential(layers)
    return mlp(jnp.array([t]))


def encoder_forward(t, latent_dim, hidden_sizes):
    # Assertion: The last element of hidden_sizes must be 2*latent_dim
    assert hidden_sizes[-1] == 2 * latent_dim, \
        f"Encoder: Last hidden_size must be {2 * latent_dim}, got {hidden_sizes[-1]}"
    
    layers = []
    # Input(1) -> first hidden layer
    layers.append(hk.Linear(hidden_sizes[0]))
    layers.append(jax.nn.relu)
    
    # Remaining hidden layers
    for i in range(len(hidden_sizes) - 1):
        layers.append(hk.Linear(hidden_sizes[i+1]))
        if i < len(hidden_sizes) - 2: # Keep activations between hidden layers
            layers.append(jax.nn.relu)
            
    mlp = hk.Sequential(layers)
    return mlp(jnp.array([t]))

def decoder_forward(z, latent_dim, hidden_sizes):
    # Assertion: latent_dim must be the first dimension of the hidden sequence
    # (or you can assert that the input size matches your expectation)
    assert len(hidden_sizes) > 0, "Decoder: hidden_sizes list cannot be empty."
    # Note: If hidden_sizes[0] is the input dimension, we validate against latent_dim:
    assert hidden_sizes[0] == latent_dim, \
        f"Decoder: First hidden_size must be {latent_dim}, got {hidden_sizes[0]}"
    
    layers = []
    # Process through layers
    for i in range(len(hidden_sizes) - 1):
        layers.append(hk.Linear(hidden_sizes[i+1]))
        layers.append(jax.nn.relu)
        
    # Final output layer to 1
    layers.append(hk.Linear(1))
    
    mlp = hk.Sequential(layers)
    return mlp(z)