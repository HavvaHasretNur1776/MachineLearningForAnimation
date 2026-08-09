import gpflow
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from gpflow.utilities import positive

class DoubleIntegratedStochasticForce(gpflow.kernels.Kernel):
    def __init__(self, variance=1.0):
        super().__init__(active_dims=[0])
        self.variance = gpflow.Parameter(variance, transform=positive())

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X

        # GPflow passes X as [N, 1] and X2 as [M, 1].
        # 1. Safely flatten them to 1D arrays: [N] and [M]
        X_flat = tf.squeeze(X, axis=-1)
        X2_flat = tf.squeeze(X2, axis=-1)

        # 2. Expand dimensions to create a clean [N, M] broadcasting grid
        X_grid = tf.expand_dims(X_flat, 1)   # Shape: [N, 1]
        X2_grid = tf.expand_dims(X2_flat, 0) # Shape: [1, M]

        # 3. Calculate the double-integral
        t_min = tf.minimum(X_grid, X2_grid)
        abs_diff = tf.abs(X_grid - X2_grid)
        
        term1 = (t_min ** 3) / 3.0
        term2 = (t_min ** 2) * abs_diff / 2.0
        
        # The result is naturally a 2D [N, M] matrix, no squeezing needed at the end!
        return self.variance * (term1 + term2)

    def K_diag(self, X):
        # The diagonal computation also needs the flattened input
        X_flat = tf.squeeze(X, axis=-1)
        return self.variance * (X_flat ** 3) / 3.0

# 2. Simulate the Physics (Hidden Oscillation)
def true_position(t, v0=0.5):
    w = 2.0 * np.pi * 1.0 
    force_amplitude = 15.0 
    t_onset = 1.0  # The exact second the particle hits the anomaly zone
    
    # 1. Standard linear inertia (particle moving in a straight line)
    base_position = v0 * t
    
    # 2. The physics of the force turning on at t_onset
    # We use np.where to apply zero force before t=1.0.
    # The linear term (t - t_onset) is the constant of integration that 
    # guarantees the particle doesn't glitch or snap its velocity upon entry.
    integrated_force = np.where(
        t < t_onset,
        0.0,
        (force_amplitude / w) * (t - t_onset) - (force_amplitude / w**2) * np.sin(w * (t - t_onset))
    )
    
    return base_position + integrated_force

np.random.seed(42)
t_train = np.linspace(0.1, 2.0, 60)[:, None]
noise_level = 0.02
x_train = true_position(t_train) + noise_level * np.random.randn(60, 1)

# 3. Build the Latent Force Model
# We combine our custom LFM kernel with a Linear kernel (for initial momentum)
kinematic_kernel = gpflow.kernels.Linear() + DoubleIntegratedStochasticForce(variance=0.1)

model = gpflow.models.GPR(
    data=(t_train, x_train), 
    kernel=kinematic_kernel, 
    mean_function=None
)

# 4. Optimize the Model
opt = gpflow.optimizers.Scipy()
opt.minimize(model.training_loss, model.trainable_variables, options=dict(maxiter=1500))

# 5. Make Predictions
t_test = np.linspace(0.1, 2.5, 300)[:, None] # Extrapolating slightly past the data
mean, var = model.predict_f(t_test)

mean = mean.numpy().flatten()
var = var.numpy().flatten()
lower = mean - 1.96 * np.sqrt(var)
upper = mean + 1.96 * np.sqrt(var)

# 6. Plot the Reconstructed Trajectory
plt.figure(figsize=(12, 6))
plt.scatter(t_train, x_train, color='red', s=20, label='Noisy Detector Hits', zorder=3)
plt.plot(t_test, true_position(t_test), color='blue', linestyle='--', label='True Physical Trajectory')
plt.plot(t_test, mean, color='green', linewidth=2, label='Agnostic LFM Reconstruction')
plt.fill_between(t_test.flatten(), lower, upper, color="green", alpha=0.2, label="95% CI")
plt.title('Agnostic LFM Trajectory Reconstruction (Double-Integrated Prior)')
plt.xlabel('Time (s)')
plt.ylabel('Position X (m)')
plt.legend()
plt.show()

import pandas as pd

# Package the time, mean prediction, and confidence intervals into a DataFrame
export_data = pd.DataFrame({
    'time': t_test.flatten(),
    'x_mean': mean,
    'ci_lower': lower,
    'ci_upper': upper,
    'true_x': true_position(t_test).flatten() # Optional: good for visual comparison
})

# Save to CSV
export_data.to_csv('agnostic_lfm_trajectory.csv', index=False)
print("Successfully exported GP data to agnostic_lfm_trajectory.csv!")
