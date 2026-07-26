# small project to test out gpflow and see if it can be used to model a bouncing ball

import gpflow
import numpy as np
import matplotlib.pyplot as plt

# define the bouncing ball function
#def bouncing_ball(t, h0=1.0, g=9.81): NO BOUNCING!
    # h0: initial height
    # g: acceleration due to gravity
    # t: time
    # returns the height of the ball at time t
#    h = h0 - 0.5 * g * t**2
#    return np.maximum(h, 0)  # the ball cannot go below ground level

#def bouncing_ball(t): approximation of falling and bouncing with decay function
    # The absolute value creates the sharp bounce impact
    # The exponential decay simulates loss of momentum over time
#    return np.abs(np.cos(t * 4.0)) * np.exp(-t * 0.8)

#def exact_physics_bouncing_ball(t_array, h0=1.0, g=9.81, e=0.75):
def bouncing_ball(t_array, h0=1.0, g=9.81, e=0.75):
    """
    h0: initial height (m)
    g: acceleration due to gravity (m/s^2)
    e: coefficient of restitution (elasticity, 0 < e < 1)
    """
    heights = []
    
    for t in t_array:
        # Calculate time and height for piecewise parabolas
        # Reset velocity on impact: v_new = e * v_impact
        t_curr = t
        h = h0
        v = 0.0 # dropped from rest
        
        # Simple Euler step to evaluate exact kinematic trajectory
        dt = 0.001
        sim_t = 0.0
        while sim_t < t:
            v -= g * dt
            h += v * dt
            
            # Ground collision
            if h <= 0:
                h = 0
                v = -v * e # Instantaneous velocity reversal with energy loss
            
            sim_t += dt
            
        heights.append(h)
        
    return np.array(heights)

## THIS is oNLY for (wrapper) to be able to avoid crashes bw python and TensorFlow
import tensorflow as tf

class PhysicsMean(gpflow.mean_functions.MeanFunction):
    def __init__(self, h0=1.0, g=9.81, e=0.75):
        super().__init__()
        # Store the physics parameters
        self.h0 = h0
        self.g = g
        self.e = e

    def __call__(self, X):
        # GPFlow passes X as a TensorFlow tensor of shape [N, 1].
        # We use tf.numpy_function to safely pass this back to your Python Euler simulation.
        def numpy_physics(t_tensor):
            # Flatten the tensor to a standard 1D numpy array
            t_array = t_tensor.flatten()
            
            # Call your exact Newtonian physics function
            heights = bouncing_ball(t_array, h0=self.h0, g=self.g, e=self.e)
            
            # Return it reshaped back to [N, 1] for GPFlow
            return heights.reshape(-1, 1).astype(np.float64)
            
        # Execute the wrapper
        res = tf.numpy_function(numpy_physics, [X], tf.float64)
        res.set_shape(X.shape)
        return res

# generate some training data
t_train = np.linspace(0, 1.5, 20)  # time
h_train = bouncing_ball(t_train) + 0.05 * np.random.randn(len(t_train))  # add some noise

# reshape the data for gpflow
t_train = t_train[:, None]  # make it a column vector
h_train = h_train[:, None]  # make it a column vector

# define the kernel and model
#kernel = gpflow.kernels.SquaredExponential()-> this is not good since it asssumes infinetly diff. functions and bouncing ball is NOT.
#kernel = gpflow.kernels.Matern12()#Matern32()
# Or, try Matern12() for an even sharper, jagged response
# SINGLE KERNEL CANNOT MODEL BOUNCING !
# 1. Define the constituent kernels for each smooth arc
# Matern32 provides the perfect amount of smoothness for a gravitational parabola
## this is also problem while connecting different regions it creates artificial jumps in mid air.
'''k1 = gpflow.kernels.Matern32()
k2 = gpflow.kernels.Matern32()
k3 = gpflow.kernels.Matern32()

# 2. Define the exact moments of impact (the change points)
impact_times = [0.453, 1.145] 

# 3. Create the Change-Point kernel
# 'steepness' controls how instantaneous the transition is. 
# A high value (e.g., 500.0) creates a hard, immediate bounce.
kernel = gpflow.kernels.ChangePoints(
    kernels=[k1, k2, k3],
    locations=impact_times,
    steepness=500.0
)
model = gpflow.models.GPR(data=(t_train, h_train), kernel=kernel, mean_function=None)
'''
# We needed to modify mean function with physics data becuase in default model mean function was zero which is kinematically wrong for bouncing ball
# 1. Instantiate your custom physics prior
mean_fn = PhysicsMean(h0=1.0, g=9.81, e=0.75)

# 2. Use a simple, smooth kernel to model ONLY the noise
# We set a small lengthscale so the GP doesn't over-smooth the noise across the impacts
kernel = gpflow.kernels.SquaredExponential(lengthscales=0.05)

# 3. Build the GPR model, passing in the custom mean
model = gpflow.models.GPR(
    data=(t_train, h_train), 
    kernel=kernel, 
    mean_function=mean_fn
)
# optimize the model
opt = gpflow.optimizers.Scipy()
opt.minimize(model.training_loss, model.trainable_variables, options=dict(maxiter=100))

# make predictions
t_test = np.linspace(0, 1.5, 100)[:, None]
mean, var = model.predict_f(t_test)

# Convert TF Tensors to NumPy arrays first
mean = mean.numpy().flatten()
var = var.numpy().flatten()

# Now NumPy math works seamlessly for plotting!
lower = mean - 1.96 * np.sqrt(var)
upper = mean + 1.96 * np.sqrt(var)

# plot the results
plt.figure(figsize=(10, 6))
# plot training data
plt.scatter(t_train, h_train, color='red', label='Training Data')
# plot true function
plt.plot(t_test, bouncing_ball(t_test), color='blue', label='True Function')
# plot GP predictions
plt.plot(t_test, mean, color='green', label='GP Mean Prediction')
# plot confidence intervals
plt.fill_between(
    t_test.flatten(),
    lower,
    upper,
    color="C0",
    alpha=0.3,
    label="95% CI"
)
plt.title('Gaussian Process Regression on Bouncing Ball Data')
plt.xlabel('Time (s)')
plt.ylabel('Height (m)')
plt.legend()
plt.show()

import csv
import os

# Define where you want to save the file
output_file = "bouncing_ball_gp_data.csv"

# Write the predicted time and height to a CSV
with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['time', 'height']) # Header row
    for t, h in zip(t_test.flatten(), mean):
        writer.writerow([t, h])

print(f"Successfully exported geometry data to {os.path.abspath(output_file)}")