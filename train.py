
from src.load_data import load_trajectories

# --- Usage ---
trajectories_loaded = load_trajectories("trajectories.csv")

print(f"Loaded trajectories shape: {trajectories_loaded.shape}")

first_traj = trajectories_loaded[0]