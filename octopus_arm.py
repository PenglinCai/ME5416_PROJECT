import numpy as np
import elastica as ea
import pickle

# Define simulator
class FlexibleOctopusArmSimulator(
    ea.BaseSystemCollection, ea.Constraints, ea.Forcing, ea.CallBacks
):
    pass

# ---- Scheduled PID Controller ----
class ScheduledPIDControlForces(ea.NoForces):
    def __init__(self, rod, kp, ki, kd, targets, segment_time, cycles, max_force=None, **kwargs):
        super().__init__(**kwargs)
        self.rod = rod
        self.kp, self.ki, self.kd = kp, ki, kd
        self.targets = [np.array(t) for t in targets]
        self.segment_time = segment_time
        self.cycles = cycles
        self.max_force = np.array(max_force) if max_force is not None else None

        # Number of phases and total simulation duration
        self.num_phases = len(self.targets)
        self.total_duration = self.segment_time * self.num_phases * self.cycles
        # Integral terms for end-effector and intermediate nodes
        self.integral_end = np.zeros(3)
        n_nodes = rod.n_elems + 1
        self.integral_y = np.zeros(n_nodes)
        self._last_time = 0.0

    def apply_forces(self, system, time):
        # Do nothing if simulation time exceeds total duration
        if time > self.total_duration:
            return
        dt = time - self._last_time
        self._last_time = time

        # Determine current phase based on scheduled timing
        cycle_dur = self.segment_time * self.num_phases
        t_in_cycle = time % cycle_dur
        phase = int(t_in_cycle // self.segment_time)
        target = self.targets[phase]

        # End-effector full-dimensional PID control
        end_pos = self.rod.position_collection[:, -1]
        end_vel = self.rod.velocity_collection[:, -1]
        err_end = target - end_pos
        self.integral_end += err_end * dt
        derror_end = end_vel
        force_end = (self.kp * err_end + self.ki * self.integral_end - self.kd * derror_end)
        if self.max_force is not None:
            force_end = np.clip(force_end, -self.max_force, self.max_force)
        system.external_forces[:, -1] += force_end

        # Intermediate nodes Y-direction PID control (every 5th node)
        base_y = self.rod.position_collection[1, 0]
        n_nodes = self.rod.n_elems + 1
        for i in range(n_nodes):
            if i % 5 != 0:
                continue
            y_i = self.rod.position_collection[1, i]
            vy_i = self.rod.velocity_collection[1, i]
            alpha = i / (n_nodes - 1)
            desired_y = base_y * (1 - alpha) + target[1] * alpha
            err_y = desired_y - y_i
            self.integral_y[i] += err_y * dt
            fy = (self.kp * err_y + self.ki * self.integral_y[i] - self.kd * vy_i)
            if self.max_force is not None:
                fy = np.clip(fy, -self.max_force[1], self.max_force[1])
            system.external_forces[1, i] += fy

# ---- Anisotropic underwater drag + buoyancy ----
class UnderwaterHydroForces(ea.NoForces):
    def __init__(self, rod, fluid_density, Cd_normal=1.2, Cd_tangential=0.1,
                 buoyancy=True, gravity=9.81, **kwargs):
        super().__init__(**kwargs)
        self.rod = rod
        # Fluid properties and drag coefficients
        self.rho_f = fluid_density
        self.Cn = Cd_normal
        self.Ct = Cd_tangential
        # Geometry for drag and buoyancy calculations
        self.radius = rod.radius.copy()
        self.diameter = 2.0 * self.radius
        self.seg_len = rod.rest_lengths.copy()
        self.seg_vol = np.pi * (self.radius**2) * self.seg_len
        # Buoyancy settings
        self.buoyancy = buoyancy
        self.g = gravity
        self.rho_rod = rod.density

    def apply_forces(self, system, time):
        pos = self.rod.position_collection
        vel = self.rod.velocity_collection
        n_elem = self.rod.n_elems

        for e in range(n_elem):
            p1, p2 = pos[:, e], pos[:, e+1]
            v1, v2 = vel[:, e], vel[:, e+1]
            seg = p2 - p1
            L = np.linalg.norm(seg)
            if L < 1e-12:
                continue
            t_hat = seg / L
            v_seg = 0.5 * (v1 + v2)
            v_para = np.dot(v_seg, t_hat) * t_hat
            v_perp = v_seg - v_para

            # Compute drag forces
            A = self.diameter[e] * L
            F_para = -0.5 * self.rho_f * self.Ct * A * np.linalg.norm(v_para) * v_para
            F_perp = -0.5 * self.rho_f * self.Cn * A * np.linalg.norm(v_perp) * v_perp
            F_drag = F_para + F_perp

            # Compute buoyancy and net weight force
            if self.buoyancy:
                rho_r = self.rho_rod[e] if hasattr(self.rho_rod, '__len__') else self.rho_rod
                W_net = (rho_r - self.rho_f) * self.seg_vol[e] * self.g
                F_bw = np.array([0.0, W_net, 0.0])
            else:
                F_bw = np.zeros(3)

            # Total hydrodynamic force on segment
            F_tot = F_drag + F_bw
            system.external_forces[:, e]     += 0.5 * F_tot
            system.external_forces[:, e + 1] += 0.5 * F_tot

# Toggle saving results
SAVE_RESULTS = True
octopus_arm_sim = FlexibleOctopusArmSimulator()

# Time scheduling control parameters
segment_time = 4
targets = [[2.0, 0.5, 0.0], [0.5, 0.5, 0.0]]
cycles = 2
final_time = segment_time * len(targets) * cycles

# CosseratRod parameters
n_elem = 100
start = np.zeros(3)
direction = np.array([0.0, 0.0, 1.0])
normal = np.array([1.0, 0.0, 0.0])
base_length = 2.3
r_base, r_tip = 0.05, 0.007
radius_nodes = np.linspace(r_base, r_tip, n_elem+1)
radius_mean = (radius_nodes[:-1] + radius_nodes[1:]) / 2

density = 1100.0
youngs_modulus = 20000
poisson_ratio = 0.5

# Create Cosserat rod
octopus_arm_rod = ea.CosseratRod.straight_rod(
    n_elem, start, direction, normal, base_length,
    base_radius=radius_mean.copy(),
    density=density,
    youngs_modulus=youngs_modulus,
    shear_modulus=youngs_modulus/(poisson_ratio+1.0)
)
octopus_arm_sim.append(octopus_arm_rod)

# Root hinge constraint
class HingeBC(ea.ConstraintBase):
    def __init__(self, fixed_position, fixed_directors, **kwargs):
        super().__init__(**kwargs)
        self.fixed_position = np.array(fixed_position)

    def constrain_values(self, rod, time):
        # Constrain base position of the rod
        rod.position_collection[..., 0] = self.fixed_position

    def constrain_rates(self, rod, time):
        # Constrain base velocity of the rod
        rod.velocity_collection[..., 0] = 0.0

octopus_arm_sim.constrain(octopus_arm_rod).using(
    HingeBC, constrained_position_idx=(0,), constrained_director_idx=(0,)
)

# Add forcing to the rod
octopus_arm_sim.add_forcing_to(octopus_arm_rod).using(
    ScheduledPIDControlForces,
    rod=octopus_arm_rod,
    kp=5, ki=0.05, kd=4.0,
    targets=targets,
    segment_time=segment_time,
    cycles=cycles,
    max_force=[4.0, 4.0, 4.0]
)
octopus_arm_sim.add_forcing_to(octopus_arm_rod).using(
    UnderwaterHydroForces,
    rod=octopus_arm_rod,
    fluid_density=1000.0,
    Cd_normal=1.2, Cd_tangential=0.1,
    buoyancy=True, gravity=9.81
)

# Callback and diagnostics recording
# Compute integration parameters and record steps

dl = base_length / n_elem
dt = 0.0007 * dl
total_steps = int(final_time / dt)
step_skip = max(1, int(total_steps / 200))
recorded_history = ea.defaultdict(list)

class OctopusArmCallBack(ea.CallBackBaseClass):
    def __init__(self, step_skip, callback_params, segment_time, targets, rod):
        super().__init__()
        self.every = step_skip
        self.callback_params = callback_params
        self.segment_time = segment_time
        self.targets = [np.array(t) for t in targets]
        self.num_phases = len(self.targets)
        self.cycle_dur = self.segment_time * self.num_phases
        self.rod = rod

    def make_callback(self, system, time, current_step):
        if current_step % self.every != 0:
            return
        # Record original data
        self.callback_params['time'].append(time)
        self.callback_params['position'].append(system.position_collection.copy())
        self.callback_params['directors'].append(system.director_collection.copy())
        self.callback_params['radius'].append(system.radius.copy())
        if time > 0.0:
            self.callback_params['internal_stress'].append(system.internal_stress.copy())
            self.callback_params['internal_couple'].append(system.internal_couple.copy())

        # Record velocity, tip velocity, tip force, and tracking error
        vel = system.velocity_collection.copy()
        self.callback_params['velocity'].append(vel)
        tip_vel = vel[:, -1]
        self.callback_params['tip_velocity'].append(tip_vel)
        force_end = system.external_forces[:, -1].copy()
        self.callback_params['force_end'].append(force_end)

        # Compute and record tracking error at end-effector
        t_in_cycle = time % self.cycle_dur
        phase = int(t_in_cycle // self.segment_time)
        target = self.targets[phase]
        tip_pos = system.position_collection[:, -1]
        err_end = target - tip_pos
        self.callback_params['error_end'].append(err_end)

octopus_arm_sim.collect_diagnostics(octopus_arm_rod).using(
    OctopusArmCallBack,
    step_skip=step_skip,
    callback_params=recorded_history,
    segment_time=segment_time,
    targets=targets,
    rod=octopus_arm_rod
)

# Finalize simulation and integrate using Position Verlet
octopus_arm_sim.finalize()
ea.integrate(ea.PositionVerlet(), octopus_arm_sim, final_time, total_steps)

# Save results to file
if SAVE_RESULTS:
    with open('octopus_arm.dat', 'wb') as f:
        pickle.dump(recorded_history, f)
