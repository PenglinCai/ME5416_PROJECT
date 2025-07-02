import pickle
import numpy as np
import matplotlib.pyplot as plt

# 1. Load data
with open('octopus_arm.dat', 'rb') as f:
    data = pickle.load(f)

# 2. Extract and stack arrays
time       = np.array(data['time'])                      # (M,)
force_end  = np.stack(data['force_end'], axis=0)         # (M,3)
error_end  = np.stack(data['error_end'], axis=0)         # (M,3)
tip_vel    = np.stack(data['tip_velocity'], axis=0)      # (M,3)

# 3. Compute norms
force_norm = np.linalg.norm(force_end, axis=1)           # (M,)
err_norm   = np.linalg.norm(error_end, axis=1)
vel_norm   = np.linalg.norm(tip_vel, axis=1)

# 4. Compute statistics including min, max, and RMSE
def compute_stats(x, name):
    rmse    = np.sqrt(np.mean(x**2))
    max_val = np.max(x)
    min_val = np.min(x)
    print(f"{name}: RMSE = {rmse:.4f}, Max = {max_val:.4f}, Min = {min_val:.4f}")

compute_stats(force_norm, "End-Force Norm")
compute_stats(err_norm,   "Tip-Error Norm")
compute_stats(vel_norm,   "Tip-Velocity Norm")

# 5. Visualization
plt.figure()
plt.plot(time, err_norm, label='Error (m)')
plt.plot(time, force_norm, label='Force (N)')
plt.plot(time, vel_norm, label='Velocity (m/s)')
plt.xlabel('Time (s)')
plt.legend()
plt.title('Tip Metrics vs Time')
plt.grid(True)
plt.tight_layout()
plt.show()
