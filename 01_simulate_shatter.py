import numpy as np
import pandas as pd
from scipy.spatial import Voronoi

def generate_target_geometry(num_shards=60, seed=42):
    """
    Generates a 3D target body bounded in a [-1, 1] cube.
    Creates Voronoi cell seed points representing shard centers.
    """
    np.random.seed(seed)
    # Target body center points (rest positions)
    rest_positions = np.random.uniform(-0.8, 0.8, size=(num_shards, 3))
    return rest_positions

def simulate_impact(rest_positions, epicenter, force_vector, t_impact=0.5, total_time=2.0, num_steps=100):
    """
    Simulates linear trajectory before and after a catastrophic impact.
    """
    num_shards = len(rest_positions)
    time_steps = np.linspace(0, total_time, num_steps)
    dt = time_steps[1] - time_steps[0]
    
    # Calculate displacement vector from epicenter to each shard
    delta_p = rest_positions - epicenter
    distances = np.linalg.norm(delta_p, axis=1, keepdims=True)
    
    # Force attenuation based on inverse-square distance from impact epicenter
    attenuation = 1.0 / (1.0 + 2.0 * distances**2)
    
    # Initial velocity imparted on shards (proportional to force direction and radial expansion)
    radial_dir = delta_p / (distances + 1e-6)
    direction = 0.6 * radial_dir + 0.4 * (force_vector / np.linalg.norm(force_vector))
    
    impulse_velocity = direction * attenuation * np.linalg.norm(force_vector)
    
    # Arrays to store trajectory data: [num_shards, num_steps, 3]
    positions = np.zeros((num_shards, num_steps, 3))
    
    for i in range(num_shards):
        p0 = rest_positions[i]
        v = np.zeros(3)
        for idx, t in enumerate(time_steps):
            if t >= t_impact:
                if idx > 0 and time_steps[idx-1] < t_impact:
                    # Apply instantaneous velocity change at t_impact
                    v = impulse_velocity[i]
            
            # Update position (simple kinematic integration)
            if idx == 0:
                positions[i, idx] = p0
            else:
                positions[i, idx] = positions[i, idx-1] + v * dt

    return time_steps, positions

# Configuration Parameters
NUM_SHARDS = 50
TRUE_EPICENTER = np.array([0.2, -0.3, 0.1])
TRUE_FORCE = np.array([12.0, 8.0, 15.0]) # N
IMPACT_TIME = 0.4 # seconds

# Run Simulation
rest_pos = generate_target_geometry(num_shards=NUM_SHARDS)
time_array, pos_array = simulate_impact(
    rest_pos, TRUE_EPICENTER, TRUE_FORCE, t_impact=IMPACT_TIME, total_time=2.0, num_steps=120
)

# Add Gaussian Noise to emulate real-world tracking sensors
NOISE_LEVEL = 0.015
noisy_pos_array = pos_array + np.random.normal(0, NOISE_LEVEL, size=pos_array.shape)

# Package into DataFrame for Export
rows = []
for shard_id in range(NUM_SHARDS):
    for t_idx, t in enumerate(time_array):
        rows.append({
            'shard_id': shard_id,
            'time': t,
            'rest_x': rest_pos[shard_id, 0],
            'rest_y': rest_pos[shard_id, 1],
            'rest_z': rest_pos[shard_id, 2],
            'pos_x': noisy_pos_array[shard_id, t_idx, 0],
            'pos_y': noisy_pos_array[shard_id, t_idx, 1],
            'pos_z': noisy_pos_array[shard_id, t_idx, 2],
            'true_x': pos_array[shard_id, t_idx, 0],
            'true_y': pos_array[shard_id, t_idx, 1],
            'true_z': pos_array[shard_id, t_idx, 2],
        })

df = pd.DataFrame(rows)
df.to_csv('shatter_trajectories.csv', index=False)
print(f"Successfully exported {len(df)} simulation records to 'shatter_trajectories.csv'")