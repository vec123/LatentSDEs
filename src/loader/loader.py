import pandas as pd
import jax.numpy as jnp

def load_trajectories(filepath="trajectories.csv"):
    """
    Loads trajectories from CSV and returns them as a JAX array.
    Shape: (N, steps, 2)
    """
    df = pd.read_csv(filepath)
    
    # Ensure data is sorted by trajectory and time
    df = df.sort_values(by=["trajectory_id", "time_step"])
    
    # Get unique IDs and steps
    n_trajectories = df["trajectory_id"].nunique()
    steps_per_traj = df["time_step"].nunique()
    
    # Extract coordinate data and reshape
    # Pivot or reshape based on the stored ID and step columns
    trajs = df[["x1", "x2"]].values.reshape(n_trajectories, steps_per_traj, 2)
    
    return jnp.array(trajs)
