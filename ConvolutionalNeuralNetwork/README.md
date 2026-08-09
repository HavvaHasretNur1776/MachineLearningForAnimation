# Neural Kinematic Reconstruction & Digital Twin Visualization
## Reversing the arrow of time
This repository contains a  computational pipeline that utilizes a 1D Convolutional Neural Network (CNN) to reconstruct the origin state of shattered kinematic debris, visualized procedurally as a reverse-entropy digital twin.
The project demonstrates how machine learning can be leveraged as a highly parallelized alternative to traditional computationally heavy likelihood frameworks for simultaneous many-body tracking and spatial reconstruction.
### Pipeline Architecture
1. Kinematic Data Generation (Python)Simulates a volumetric explosion of 50 rigid bodies.
    - Calculates discrete time-series trajectories, including positional displacement and rotational momentum over 120 frames.
    - Exports raw telemetry data as a chaotic, unorganized point cloud sequence.
3. Temporal Machine Learning (TensorFlow / Keras)
    - Implements a 1D CNN to process the temporal kinematic data.
    - The network acts as an automated feature extractor, sliding across the time-series data $x_t$ to implicitly learn spatial derivatives like velocity and acceleration.
    - Bypasses the complexity bottlenecks found in standard regressions by processing the simultaneous chaos of all tracking points in parallel.
    - Outputs predicted Cartesian coordinates for the object's rest state and a normalized reconstruction_error metric representing the model's localized uncertainty.
4. Procedural Visualization & Error Mapping (SideFX Houdini / VEX)
   - Ingests the static ML predictions and dynamic telemetry into a procedural node network.
   - Physical Simulation: Reverses the time of the dynamic data, applying displacement vectors to a 3D geometry to simulate the shattered pieces going back in time for physical reformation of the original object.
   - Digital Twin Hologram: Constructs a real-time architectural wireframe representing the neural network's structural prediction.
   - Procedural Diagnostics: Utilizes custom VEX code to read the CNN's reconstruction_error and visually glitch the wireframe geometry in exact spatial zones where the model struggled with pattern recognition.
<img width="800" height="449" alt="ezgif com-video-to-gif-converter" src="https://github.com/user-attachments/assets/7ff01fd0-92f2-4ba5-81bc-96b6b0b007ef" />
