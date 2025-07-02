# Create and activate the conda environment:
conda create -n ME5416_Project python=3.11 -y
conda activate ME5416_Project

# Install core numerical packages via conda:
   conda install -c conda-forge numpy scipy tqdm -y

# Install PyElastica with pip:
   pip install pyelastica

# Install system dependencies for rendering:
   sudo apt install povray ffmpeg

# Octopus Arm Simulation

This project simulates a flexible, octopus‑like arm in an underwater environment using **PyElastica**, applies a scheduled PID controller and hydrodynamic forces, and provides tools for evaluation and rendering.

## File Structure

- **octopus_arm.py**  
  Runs the main simulation, recording time, position, radii, forces, errors, etc., into "octopus_arm.dat".

- **eval.py**  
  Loads "octopus_arm.dat", computes RMSE and maximum values for force, error, and velocity norms, and plots these metrics over time.

- **render.py**  
  Reads "octopus_arm.dat", generates POV‑Ray scripts for each frame and camera view, renders frames in parallel, and assembles them into videos using ffmpeg.


