import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

print("Loading shatter trajectory data...")
# 1. Load the Data
df = pd.read_csv('shatter_trajectories.csv')

# 2. Preprocess Data into Time-Series Tensors
num_shards = df['shard_id'].nunique()
num_steps = df[df['shard_id'] == 0].shape[0]

# X will be our input: Noisy [x, y, z] over time
X = np.zeros((num_shards, num_steps, 3))
# Y will be our target: True rest position [x, y, z]
Y = np.zeros((num_shards, 3))

for shard_id in range(num_shards):
    shard_data = df[df['shard_id'] == shard_id].sort_values('time')
    X[shard_id] = shard_data[['pos_x', 'pos_y', 'pos_z']].values
    # Rest position is static, just take the first row's value
    Y[shard_id] = shard_data[['rest_x', 'rest_y', 'rest_z']].iloc[0].values

# 3. Build the Temporal Pattern Recognition Model (1D CNN)
def build_reconstruction_model(input_shape):
    model = models.Sequential([
        # Extract local kinetic features (velocity, acceleration) from the sequence
        layers.Conv1D(64, kernel_size=5, activation='relu', input_shape=input_shape),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(128, kernel_size=3, activation='relu'),
        
        # Collapse the temporal sequence into a singular spatial understanding
        layers.GlobalAveragePooling1D(),
        
        # Regress to the final 3D coordinate
        layers.Dense(64, activation='relu'),
        layers.Dense(3, name='predicted_rest_position')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
                  loss='mse',
                  metrics=['mae'])
    return model

print("Building and training the Neural Network...")
model = build_reconstruction_model(input_shape=(num_steps, 3))

# Train the model (Using a small epoch count for quick iteration)
history = model.fit(X, Y, epochs=150, batch_size=16, verbose=0)
print(f"Final Training Loss (MSE): {history.history['loss'][-1]:.4f}")

# 4. Generate Predictions and Calculate Confidence/Error
print("Generating Digital Twin predictions...")
predictions = model.predict(X)

# Calculate spatial Euclidean error for each shard
# This will act as our 'uncertainty' metric to drive Houdini glitches
errors = np.linalg.norm(predictions - Y, axis=1)

# Normalize errors between 0 and 1 for easier shading in Houdini
max_error = np.max(errors)
normalized_errors = errors / (max_error + 1e-6)

# 5. Export Results for Houdini
export_rows = []
for i in range(num_shards):
    export_rows.append({
        'shard_id': i,
        'pred_x': predictions[i, 0],
        'pred_y': predictions[i, 1],
        'pred_z': predictions[i, 2],
        'true_rest_x': Y[i, 0],
        'true_rest_y': Y[i, 1],
        'true_rest_z': Y[i, 2],
        'reconstruction_error': normalized_errors[i]  # 0 = Perfect, 1 = Max Glitch
    })

results_df = pd.DataFrame(export_rows)
results_df.to_csv('ml_reconstruction_results.csv', index=False)
print("Successfully exported network predictions to 'ml_reconstruction_results.csv'")