# Modeling 3D Kinematics with Gaussian Process Regression
This toy project explores the use of Gaussian Process Regression (GPR) via GPflow to model and predict the physical trajectory of an object in different physical environments to be 3D animated in Houdini.

## Into the Unknown Force
This project reconstructs a particle's trajectory from noisy spatial coordinates under the influence of an unknown force using Gaussian Process Regression (GPR) and visualizes the inferred kinetic anomaly using procedural vector fields in SideFX Houdini.
### Agnostic Latent Force Models (LFM)
Instead of assuming a parametric physical model for the unknown force $f(t)$, we go with complete model-agnostic appriach and, we place a zero-mean Gaussian Process prior on it:
- $f(t) \sim \mathcal{GP}(0, k_f(t, t'))$
Because acceleration is the second derivative of position ($\ddot{x}(t) = f(t)$), we do not need to compute the force explicitly. Instead, we apply a double linear operator to the covariance function to model the spatial position $x(t)$ directly:
- $k_x(t, t') = \int_0^t \int_0^{t'} k_f(u, v) \,du \,dv$
This allows the model to agnostically infer trajectory disruptions and 95% confidence intervals entirely from observational data, without knowing the underlying physics of the anomaly.

<img width="600" height="400" alt="Screenshot 2026-07-27 at 15 12 27" src="https://github.com/user-attachments/assets/41f26daf-130f-449d-8f8c-3d17fbe0b3e9" />


<img width="650" height="350" alt="unknown_force_v2" src="https://github.com/user-attachments/assets/c14e76b7-1eb5-4ca1-9818-644e5eb99d28" />




## Bouncing Ball
Modelling a bouncing ball presents a small challenge because the system exists in two contradictory physical regimes: Airborne Flight and Ground impact


### Physics-Informed Priors
To solve this, the model leverages a Custom Mean Function. Instead of relying on a purely data-driven approach, an exact Newtonian Euler integrator was injected directly into the GP as the prior mean. 

- The Mean Function handles the strict deterministic laws of physics, ensuring perfect gravitational parabolas and sharp kinematic rebounds at the $Y=0$ boundary.
- The Kernel (Squared Exponential) is relegated entirely to modeling the stochastic noise or residuals in the positional data.

## Why GPR for 3D Motion and Kinematics?
Gaussian Processes offer powerful advantages for 3D animation and spatial tracking:

- Handling Sensor Noise: When working with raw motion capture data or noisy spatial tracking sensors, GPR excels at separating random noise from the true trajectory without destroying the underlying kinematics.
- Scattering and Boundary Events: Standard statistical models struggle with sudden shifts in momentum. Much like how tracking algorithms must account for a particle's perfectly smooth trajectory being abruptly altered by scattering inside a dense detector layer, animation systems must handle sudden velocity reversals at impact. Injecting deterministic physics into the GP allows it to handle these rigid boundary collisions flawlessly.
- Predictive Interpolation: GPR provides robust interpolation between sparse keyframes, outputting not just a mean prediction for the 3D software to render, but a fully quantified confidence interval for the geometry's motion.

<img width="500" height="300" alt="bouncing_ball-ezgif com-video-to-gif-converter" src="https://github.com/user-attachments/assets/298e1e92-1935-4e28-9b6f-b5a5242e26ea" />

<img width="600" height="400" alt="gp_bouncin_ball" src="https://github.com/user-attachments/assets/de9a305f-baea-44c4-8107-9b02181031e6" />

