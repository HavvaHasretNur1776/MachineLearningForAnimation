# Modeling 3D Kinematics with Gaussian Process Regression
This toy project explores the use of Gaussian Process Regression (GPR) via GPflow to model and predict the physical trajectory of a bouncing ball for 3D animation in Houdini.
## The Approach
Modelling a bouncing ball presents a small challenge because the system exists in two contradictory physical regimes:

- Airborne Flight: Governed by gravity, forming a perfectly smooth, infinitely differentiable quadratic curve.Ground 

- Impact: An instantaneous velocity reversal with energy loss, resulting in a sharp, non-differentiable vertex.Initially, standard stationary kernels (like Matérn 3/2 or Squared Exponential) were tested. However, these kernels enforce continuous priors across the entire timeline. When confronted with the hard physical boundary of the floor, the model either rounded off the bounce—visually causing the geometry to sink into the floor in Houdini—or created severe oscillation artifacts.

### The Solution: Physics-Informed Priors
To solve this, the model leverages a Custom Mean Function. Instead of relying on a purely data-driven approach, an exact Newtonian Euler integrator was injected directly into the GP as the prior mean. 

- The Mean Function handles the strict deterministic laws of physics, ensuring perfect gravitational parabolas and sharp kinematic rebounds at the $Y=0$ boundary.
- The Kernel (Squared Exponential) is relegated entirely to modeling the stochastic noise or residuals in the positional data.

### Why GPR for 3D Motion and Kinematics?
Gaussian Processes offer powerful advantages for 3D animation and spatial tracking:

- Handling Sensor Noise: When working with raw motion capture data or noisy spatial tracking sensors, GPR excels at separating random noise from the true trajectory without destroying the underlying kinematics.
- Scattering and Boundary Events: Standard statistical models struggle with sudden shifts in momentum. Much like how tracking algorithms must account for a particle's perfectly smooth trajectory being abruptly altered by scattering inside a dense detector layer, animation systems must handle sudden velocity reversals at impact. Injecting deterministic physics into the GP allows it to handle these rigid boundary collisions flawlessly.
- Predictive Interpolation: GPR provides robust interpolation between sparse keyframes, outputting not just a mean prediction for the 3D software to render, but a fully quantified confidence interval for the geometry's motion.

<img width="500" height="300" alt="bouncing_ball-ezgif com-video-to-gif-converter" src="https://github.com/user-attachments/assets/298e1e92-1935-4e28-9b6f-b5a5242e26ea" />

<img width="600" height="400" alt="gp_bouncin_ball" src="https://github.com/user-attachments/assets/de9a305f-baea-44c4-8107-9b02181031e6" />

