import matplotlib.pyplot as plt
import math
import random
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.optimize import linear_sum_assignment
from scipy.optimize import minimize
import time

try:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"Warning: Could not set preferred font. Error: {e}")
    print("Using default font.")
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']


class CarController:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.velocity = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.max_speed = 5.0
        self.acceleration = 3.0
        self.trajectory = []

        self.resistance_coeff = 0.8

        self.dt = 0.10
        self.prediction_horizon = 5
        self.max_accel = 3.0
        self.weight_goal = 20.0
        self.weight_velocity = 5.0
        self.weight_accel = 0.1
        self.weight_obstacle = 60.0
        self.weight_boundary = 50.0
        self.car_radius = 0.5

        self.obstacle_visual_radius = 0.3
        self.boundary_safe_distance = 1
        self.avoidance_margin = 0.2
        self.warning_zone_distance = 1.0

        self.world_size = 10

        self.blend_safe_dist = self.car_radius + self.obstacle_visual_radius + self.avoidance_margin
        self.manual_override_dist = self.blend_safe_dist + 3.0
        self.blend_alpha = 0.0

        self.radar_max_distance = 15
        self.radar_angle_res = 1
        self.radar_noise_std = 0.1
        self.raw_scan_data = []

        self.obstacle_clusters = []
        self.obstacle_history = {}
        self.max_history_length = 10
        self.max_expected_obstacle_speed = 1.0

        self.dbscan_eps = 0.5
        self.dbscan_min_samples = 3

        self.fig, self.ax = plt.subplots()
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('key_release_event', self.on_key_release)
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_click)
        self.setup_plot()

        self.control_vector = np.array([0.0, 0.0])

        self.obstacles = []
        self.init_obstacles()

        self.car_circle = self.ax.add_patch(plt.Circle((self.x, self.y), self.car_radius, color='red', zorder=5))
        self.trajectory_line, = self.ax.plot([], [], 'b-', lw=1, alpha=0.7)
        self.info_text = self.ax.text(0.05, 0.95, '', transform=self.ax.transAxes, va='top', fontsize=10)
        self.scan_scatter = self.ax.scatter(np.empty(0), np.empty(0), c='lime', s=5, marker='.', alpha=0.5)
        self.border_rect = plt.Rectangle((-self.world_size, -self.world_size), 2 * self.world_size, 2 * self.world_size, linewidth=1,
                                         edgecolor='red', linestyle='--', facecolor='none', zorder=1)
        self.ax.add_patch(self.border_rect)
        self.mpc_trajectory_line, = self.ax.plot([], [], 'g--', lw=1.5, alpha=0.8)

        self.goal = None
        self.goal_marker, = self.ax.plot([], [], 'rx', markersize=10, zorder=6)

        self._last_time = time.time()
        self._current_frame_dt = 0.0

        from matplotlib.animation import FuncAnimation
        self.ani = FuncAnimation(self.fig, self.update_frame, frames=None, interval=30, blit=False, repeat=False)

        print(f"MPC blend parameters: safe_dist={self.blend_safe_dist:.2f}m, override_dist={self.manual_override_dist:.2f}m")
        print(f"Collision/MPC avoidance minimum distance: {self.car_radius + self.obstacle_visual_radius + self.avoidance_margin:.2f}m")
        print(f"Boundary safe distance: {self.boundary_safe_distance:.2f}m")
        print(f"Resistance coefficient: {self.resistance_coeff}")
        print(f"MPC Prediction Horizon: {self.prediction_horizon} steps ({self.prediction_horizon * self.dt:.2f} seconds)")
        print(f"MPC Optimizer Max Iterations (SLSQP): 200")

    def setup_plot(self):
        self.ax.set_title("Omnidirectional Autonomous Car - Radar (Angle-Distance) MPC Obstacle Avoidance & Manual Blend\nWASD: Move  Space: Emergency Stop  Mouse Click: Set Goal\nGreen Dashed Line: MPC Predicted Path  Orange Circle: True Obstacle  Green Dot: Radar Point Cloud", fontsize=10)
        self.ax.set_xlim(-self.world_size, self.world_size)
        self.ax.set_ylim(-self.world_size, self.world_size)
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')

    def on_key_press(self, event):
        if not hasattr(self, '_pressed_keys'):
            self._pressed_keys = set()
        self._pressed_keys.add(event.key)
        self.update_control_vector()

    def on_key_release(self, event):
        if hasattr(self, '_pressed_keys') and event.key in self._pressed_keys:
            self._pressed_keys.remove(event.key)
        self.update_control_vector()

    def update_control_vector(self):
        if not hasattr(self, '_pressed_keys'):
            self._pressed_keys = set()

        vec = np.array([0.0, 0.0])
        if ' ' in self._pressed_keys or 'space' in self._pressed_keys:
            self.vx = 0.0
            self.vy = 0.0
            self.velocity = 0.0
            self.control_vector = np.array([0.0, 0.0])
            self._pressed_keys.discard(' ')
            self._pressed_keys.discard('space')
            self.goal = None
            self.goal_marker.set_data([], [])
            return

        if 'w' in self._pressed_keys or 'up' in self._pressed_keys:    vec += [0, 1]
        if 's' in self._pressed_keys or 'down' in self._pressed_keys:  vec += [0, -1]
        if 'a' in self._pressed_keys or 'left' in self._pressed_keys: vec += [-1, 0]
        if 'd' in self._pressed_keys or 'right' in self._pressed_keys: vec += [1, 0]

        if np.linalg.norm(vec) > 0:
            self.control_vector = vec / np.linalg.norm(vec)
        else:
            self.control_vector = vec

    def on_mouse_click(self, event):
        if event.inaxes == self.ax:
            self.goal = (event.xdata, event.ydata)
            self.goal_marker.set_data(self.goal[0], self.goal[1])
            print(f"Goal set to: {self.goal}")
            if hasattr(self, '_pressed_keys'):
                self._pressed_keys = set()
            self.control_vector = np.array([0.0, 0.0])

    def init_obstacles(self):
        if hasattr(self, '_obstacle_patches'):
            for patch in self._obstacle_patches:
                patch.remove()
        self._obstacle_patches = []

        self.obstacles = []
        obstacle_id_counter = 0

        true_obstacle_radius = 0.4

        for i in range(6):
            speed = random.uniform(0.3, 0.6)

            if i < 3:
                motion_type = 'circle'
                angle = random.uniform(0, 2*math.pi)
                radius = random.uniform(4, 8)
                while np.linalg.norm([radius * math.cos(angle), radius * math.sin(angle)]) < 3.0:
                    angle = random.uniform(0, 2*math.pi)
                    radius = random.uniform(4, 8)

                cx = radius * math.cos(angle)
                cy = radius * math.sin(angle)

                obstacle = {
                    'id': obstacle_id_counter,
                    'circle': plt.Circle((cx, cy), true_obstacle_radius, color='darkorange', alpha=0.8, zorder=3),
                    'motion_type': motion_type,
                    'speed': speed,
                    'angle': angle,
                    'radius': radius,
                    'true_radius': true_obstacle_radius
                }

                self.ax.add_patch(obstacle['circle'])
                self._obstacle_patches.append(obstacle['circle'])

            else:
                motion_type = 'linear'
                cx = random.uniform(-self.world_size + true_obstacle_radius, self.world_size - true_obstacle_radius)
                cy = random.uniform(-self.world_size + true_obstacle_radius, self.world_size - true_obstacle_radius)

                while np.linalg.norm([cx, cy]) < 3.0:
                    cx = random.uniform(-self.world_size + true_obstacle_radius, self.world_size - true_obstacle_radius)
                    cy = random.uniform(-self.world_size + true_obstacle_radius, self.world_size - true_obstacle_radius)

                target_x = random.uniform(-self.world_size + true_obstacle_radius, self.world_size - true_obstacle_radius)
                target_y = random.uniform(-self.world_size + true_obstacle_radius, self.world_size - true_obstacle_radius)

                obstacle = {
                    'id': obstacle_id_counter,
                    'circle': plt.Circle((cx, cy), true_obstacle_radius, color='darkorange', alpha=0.8, zorder=3),
                    'motion_type': motion_type,
                    'speed': speed,
                    'target_pos': (target_x, target_y),
                    'true_radius': true_obstacle_radius
                }
                self.ax.add_patch(obstacle['circle'])
                self._obstacle_patches.append(obstacle['circle'])

            self.obstacles.append(obstacle)
            obstacle_id_counter += 1

        print(f"Initialized {len(self.obstacles)} simulation obstacles.")

    def ray_intersect_circle(self, origin, theta_rad, circle_center_pos, circle_radius):
        dx, dy = math.cos(theta_rad), math.sin(theta_rad)
        cx, cy = circle_center_pos
        ox, oy = origin

        fx = ox - cx
        fy = oy - cy
        r = circle_radius

        A = dx*dx + dy*dy
        B = 2 * (fx * dx + fy * dy)
        C = fx * fx + fy * fy - r * r

        delta = B * B - 4 * A * C

        if delta < 0:
            return None

        sqrt_delta = math.sqrt(delta)

        t1 = (-B - sqrt_delta) / (2 * A)
        t2 = (-B + sqrt_delta) / (2 * A)

        positive_ts = [t for t in [t1, t2] if t > 1e-9]

        if not positive_ts:
            return None

        min_t = min(positive_ts)

        return min_t

    def scan_environment(self):
        self.raw_scan_data = []
        origin = (self.x, self.y)

        car_orientation_rad = math.radians(self.theta)

        for angle_deg_relative in np.arange(0, 360, self.radar_angle_res):
            angle_rad_world = car_orientation_rad + math.radians(angle_deg_relative)

            closest_dist = self.radar_max_distance
            hit_found = False

            for obstacle in self.obstacles:
                t = self.ray_intersect_circle(origin, angle_rad_world, obstacle['circle'].center, obstacle['true_radius'])

                if t is not None and t < closest_dist:
                    closest_dist = t
                    hit_found = True

            if hit_found:
                noisy_dist = closest_dist + random.gauss(0, self.radar_noise_std)
                noisy_dist = max(0.0, noisy_dist)
                noisy_dist = min(noisy_dist, self.radar_max_distance + self.radar_noise_std * 3)

                if noisy_dist <= self.radar_max_distance:
                    self.raw_scan_data.append((angle_deg_relative, noisy_dist))

    def convert_polar_to_cartesian(self, angle_deg_relative, distance, car_x, car_y, car_theta_deg):
        car_theta_rad = math.radians(car_theta_deg)
        angle_rad_relative = math.radians(angle_deg_relative)

        angle_rad_world = car_theta_rad + angle_rad_relative

        world_x = car_x + distance * math.cos(angle_rad_world)
        world_y = car_y + distance * math.sin(angle_rad_world)
        return (world_x, world_y)

    def cluster_scan_points(self):
        if not self.raw_scan_data:
            self.scan_points_for_plot = np.empty((0, 2))
            return []

        world_scan_points = [
            self.convert_polar_to_cartesian(angle, dist, self.x, self.y, self.theta)
            for angle, dist in self.raw_scan_data
            if dist <= self.radar_max_distance
        ]
        self.scan_points_for_plot = np.array(world_scan_points)

        if self.scan_points_for_plot.shape[0] < self.dbscan_min_samples:
            return []

        dbscan = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)
        clusters = dbscan.fit_predict(self.scan_points_for_plot)

        clustered_points_map = {}
        for i, label in enumerate(clusters):
            if label != -1:
                if label not in clustered_points_map:
                    clustered_points_map[label] = []
                clustered_points_map[label].append(self.scan_points_for_plot[i])

        obstacle_centers = []
        for label, points in clustered_points_map.items():
            center = np.mean(points, axis=0)
            obstacle_centers.append(center)

        return obstacle_centers

    def track_obstacles(self, current_obstacle_centers, dt):
        num_tracked = len(self.obstacle_clusters)
        num_detected = len(current_obstacle_centers)

        cost_matrix = np.full((num_tracked, num_detected), np.inf)

        max_possible_movement = self.max_expected_obstacle_speed * dt
        match_threshold = max_possible_movement + self.dbscan_eps * 1.5

        for i in range(num_tracked):
            for j in range(num_detected):
                dist = np.linalg.norm(np.array(self.obstacle_clusters[i]['center']) - np.array(current_obstacle_centers[j]))
                if dist < match_threshold:
                    cost_matrix[i, j] = dist

        updated_clusters = []
        matched_detected_indices = set()
        matched_tracked_indices = set()

        if num_tracked > 0 and num_detected > 0:
            try:
                row_ind, col_ind = linear_sum_assignment(cost_matrix)

                for i, j in zip(row_ind, col_ind):
                    if cost_matrix[i, j] < np.inf:
                        tracked_cluster = self.obstacle_clusters[i]
                        detected_center = current_obstacle_centers[j]

                        tracked_cluster['center'] = detected_center
                        tracked_cluster['age'] = 0

                        obstacle_id = tracked_cluster['id']
                        if obstacle_id not in self.obstacle_history:
                            self.obstacle_history[obstacle_id] = []
                        self.obstacle_history[obstacle_id].append(detected_center)
                        if len(self.obstacle_history[obstacle_id]) > self.max_history_length:
                            self.obstacle_history[obstacle_id].pop(0)

                        updated_clusters.append(tracked_cluster)
                        matched_tracked_indices.add(i)
                        matched_detected_indices.add(j)

            except ValueError as e:
                pass

        for j in range(num_detected):
            if j not in matched_detected_indices:
                new_center = current_obstacle_centers[j]
                existing_ids = list(self.obstacle_history.keys())
                new_id = max(existing_ids + [-1]) + 1 if existing_ids else 0
                new_cluster = {'id': new_id, 'center': new_center, 'age': 0}
                updated_clusters.append(new_cluster)
                self.obstacle_history[new_id] = [new_center]

        for i in range(num_tracked):
            if i not in matched_tracked_indices:
                lost_cluster = self.obstacle_clusters[i]
                lost_cluster.setdefault('age', 0)
                lost_cluster['age'] += 1
                if lost_cluster['age'] < 5:
                    updated_clusters.append(lost_cluster)
                else:
                    if lost_cluster['id'] in self.obstacle_history:
                        del self.obstacle_history[lost_cluster['id']]

        self.obstacle_clusters = updated_clusters

    def predict_obstacle_positions(self):
        predicted_obstacle_paths = []

        for cluster in self.obstacle_clusters:
            obstacle_id = cluster['id']
            current_pos = np.array(cluster['center'])
            predicted_path = [current_pos.tolist()]

            velocity_estimate = np.array([0.0, 0.0])

            if obstacle_id in self.obstacle_history and len(self.obstacle_history[obstacle_id]) >= 2:
                history = self.obstacle_history[obstacle_id]
                pos_latest = np.array(history[-1])
                pos_previous = np.array(history[-2])

                history_time_diff_approx = self.dt

                if history_time_diff_approx > 1e-9:
                    velocity_estimate = (pos_latest - pos_previous) / history_time_diff_approx

                current_est_speed = np.linalg.norm(velocity_estimate)
                if current_est_speed > self.max_expected_obstacle_speed:
                    if current_est_speed > 1e-9:
                        velocity_estimate = (velocity_estimate / current_est_speed) * self.max_expected_obstacle_speed
                    else:
                        velocity_estimate = np.array([0.0, 0.0])

            current_predicted_pos = current_pos.copy()
            for i in range(self.prediction_horizon):
                current_predicted_pos = current_predicted_pos + velocity_estimate * self.dt
                predicted_path.append(current_predicted_pos.tolist())

            predicted_obstacle_paths.append(predicted_path)

        return predicted_obstacle_paths

    def motion_model(self, state, control, dt):
        x, y, vx, vy = state
        ax_control, ay_control = control

        ax_resistance = -self.resistance_coeff * vx
        ay_resistance = -self.resistance_coeff * vy

        net_ax = ax_control + ax_resistance
        net_ay = ay_control + ay_resistance

        next_vx = vx + net_ax * dt
        next_vy = vy + net_ay * dt

        current_speed_sq = next_vx**2 + next_vy**2
        max_speed_sq = self.max_speed**2
        if current_speed_sq > max_speed_sq:
            if current_speed_sq > 1e-12:
                scale = self.max_speed / math.sqrt(current_speed_sq)
                next_vx *= scale
                next_vy *= scale
            else:
                next_vx = 0.0
                next_vy = 0.0

        next_x = x + next_vx * dt
        next_y = y + next_vy * dt

        return np.array([next_x, next_y, next_vx, next_vy])

    def calculate_blend_ratio(self, dist_to_nearest_obstacle):
        if dist_to_nearest_obstacle is None or dist_to_nearest_obstacle >= self.manual_override_dist:
            return 0.0
        if dist_to_nearest_obstacle <= self.blend_safe_dist:
            return 1.0

        range_dist = self.manual_override_dist - self.blend_safe_dist
        if range_dist <= 1e-9:
            return 1.0 if dist_to_nearest_obstacle <= self.blend_safe_dist else 0.0

        clamped_dist = np.clip(dist_to_nearest_obstacle, self.blend_safe_dist, self.manual_override_dist)

        alpha = (self.manual_override_dist - clamped_dist) / range_dist

        return np.clip(alpha, 0.0, 1.0)

    def mpc_cost(self, u_flat, x0, goal, predicted_obstacle_paths):
        cost = 0.0
        state = np.array(x0)

        self.mpc_trajectory = [state[:2].tolist()]

        alpha = self.blend_alpha

        obstacle_cost_weight_scaled = alpha * self.weight_obstacle
        goal_or_manual_weight_scaled = (1.0 - alpha) * (self.weight_goal if goal is not None else self.weight_velocity)

        for i in range(self.prediction_horizon):
            if (i * 2 + 1) < len(u_flat):
                control_i = np.array([u_flat[i * 2], u_flat[i * 2 + 1]])
            else:
                control_i = np.array([0.0, 0.0])

            state = self.motion_model(state, control_i, self.dt)
            x, y, vx, vy = state

            self.mpc_trajectory.append(state[:2].tolist())

            goal_or_manual_cost_step = 0.0
            if goal is not None:
                dist_to_goal = np.linalg.norm(state[:2] - goal)
                goal_or_manual_cost_step = dist_to_goal**2
            else:
                desired_velocity = self.control_vector * self.max_speed
                velocity_error_sq = np.sum((state[2:] - desired_velocity)**2)
                goal_or_manual_cost_step = velocity_error_sq

            cost += goal_or_manual_weight_scaled * goal_or_manual_cost_step

            cost += self.weight_accel * np.sum(control_i**2)

            obstacle_cost_step = 0.0
            if obstacle_cost_weight_scaled > 1e-9 and predicted_obstacle_paths:
                safe_distance = self.car_radius + self.obstacle_visual_radius + self.avoidance_margin
                warning_zone_dist_start = safe_distance
                warning_zone_dist_end = safe_distance + self.warning_zone_distance

                for obs_path in predicted_obstacle_paths:
                    if len(obs_path) > i:
                        obs_pos_at_i = np.array(obs_path[i])

                        dist_to_obstacle = np.linalg.norm(state[:2] - obs_pos_at_i)

                        if dist_to_obstacle < safe_distance:
                            penetration = safe_distance - dist_to_obstacle
                            obstacle_cost_step += penetration**2 * 100
                        elif dist_to_obstacle < warning_zone_dist_end:
                            smoothing_factor = self.warning_zone_distance / 3.0
                            obstacle_cost_step += math.exp((safe_distance - dist_to_obstacle) / smoothing_factor)

            cost += obstacle_cost_weight_scaled * obstacle_cost_step

            boundary_cost_step = 0.0
            dist_left = state[0] - (-self.world_size)
            dist_right = self.world_size - state[0]
            dist_bottom = state[1] - (-self.world_size)
            dist_top = self.world_size - state[1]

            boundary_safe_distance = self.boundary_safe_distance

            if dist_left < boundary_safe_distance:
                boundary_cost_step += (boundary_safe_distance - dist_left)**2
            if dist_right < boundary_safe_distance:
                boundary_cost_step += (boundary_safe_distance - dist_right)**2
            if dist_bottom < boundary_safe_distance:
                boundary_cost_step += (boundary_safe_distance - dist_bottom)**2
            if dist_top < boundary_safe_distance:
                boundary_cost_step += (boundary_safe_distance - dist_top)**2

            cost += self.weight_boundary * boundary_cost_step

        return cost

    def solve_mpc(self, x0, goal, predicted_obstacle_paths):
        effective_goal = np.array(goal) if goal is not None else None
        u0 = np.zeros(self.prediction_horizon * 2)
        bounds = [(-self.max_accel, self.max_accel), (-self.max_accel, self.max_accel)] * self.prediction_horizon

        objective = lambda u_flat: self.mpc_cost(u_flat, x0, effective_goal, predicted_obstacle_paths)

        result = minimize(objective, u0,
                          method='SLSQP',
                          bounds=bounds,
                          options={'maxiter': 200, 'ftol': 1e-2, 'disp': False})

        if result.success:
            return result.x
        else:
            self.mpc_trajectory = []
            return np.zeros(self.prediction_horizon * 2)

    def update_movement(self, dt):
        current_state = np.array([self.x, self.y, self.vx, self.vy])

        predicted_obstacle_paths = self.predict_obstacle_positions()

        u_flat = self.solve_mpc(current_state, self.goal, predicted_obstacle_paths)

        commanded_ax = 0.0
        commanded_ay = 0.0

        if u_flat is not None and len(u_flat) >= 2:
            commanded_ax = u_flat[0]
            commanded_ay = u_flat[1]

        resistance_ax = -self.resistance_coeff * self.vx
        resistance_ay = -self.resistance_coeff * self.vy

        net_ax = commanded_ax + resistance_ax
        net_ay = commanded_ay + resistance_ay

        self.vx += net_ax * dt
        self.vy += net_ay * dt

        current_speed = math.hypot(self.vx, self.vy)
        if current_speed > self.max_speed:
            if current_speed > 1e-12:
                scale = self.max_speed / current_speed
                self.vx *= scale
                self.vy *= scale
            else:
                self.vx = 0.0
                self.vy = 0.0

        self.velocity = math.hypot(self.vx, self.vy)

        self.x += self.vx * dt
        self.y += self.vy * dt

        self.trajectory.append((self.x, self.y))
        max_trajectory_length = 500
        if len(self.trajectory) > max_trajectory_length:
            self.trajectory.pop(0)

    def check_collision(self, obstacle_positions):
        car_pos = np.array([self.x, self.y])
        for obstacle in self.obstacles:
            obs_pos = np.array(obstacle['circle'].center)
            obs_radius = obstacle['true_radius']
            collision_distance = self.car_radius + obs_radius
            dist = np.linalg.norm(car_pos - obs_pos)
            if dist < collision_distance:
                return True
        return False

    def update_frame(self, frame=None):
        current_time = time.time()
        dt = max(0.001, current_time - self._last_time)
        self._last_time = current_time
        self._current_frame_dt = dt

        dt = min(dt, 0.1)

        for obstacle in self.obstacles:
            if obstacle['motion_type'] == 'circle':
                cx_old, cy_old = obstacle['circle'].center
                radius = obstacle['radius']

                if radius > 1e-9:
                    angular_speed_rad_per_sec = obstacle['speed'] / radius
                    angle_moved = angular_speed_rad_per_sec * dt
                    obstacle['angle'] = (obstacle['angle'] + angle_moved) % (2 * math.pi)
                    cx_new = radius * math.cos(obstacle['angle'])
                    cy_new = radius * math.sin(obstacle['angle'])
                    obstacle['circle'].center = (cx_new, cy_new)
                else:
                    obstacle['circle'].center = (0.0, 0.0)

            elif obstacle['motion_type'] == 'linear':
                target = np.array(obstacle['target_pos'])
                current = np.array(obstacle['circle'].center)
                direction = target - current
                dist_to_target = np.linalg.norm(direction)

                movement_this_step = obstacle['speed'] * dt

                if dist_to_target < max(movement_this_step * 0.5, 0.05):
                    cx, cy = target
                    obstacle['circle'].center = (cx, cy)
                    obs_radius = obstacle['true_radius']
                    new_target_x = random.uniform(-self.world_size + obs_radius, self.world_size - obs_radius)
                    new_target_y = random.uniform(-self.world_size + obs_radius, self.world_size - obs_radius)
                    obstacle['target_pos'] = (new_target_x, new_target_y)
                else:
                    if dist_to_target > 1e-9:
                        move_vec = direction / dist_to_target * movement_this_step
                        cx, cy = current + move_vec
                        obstacle['circle'].center = (cx, cy)

        self.scan_environment()
        current_obstacle_centers_world = self.cluster_scan_points()
        self.track_obstacles(current_obstacle_centers_world, dt)

        car_pos_np = np.array([self.x, self.y])
        d_min = float('inf')
        if self.obstacle_clusters:
            distances = [np.linalg.norm(car_pos_np - np.array(c['center'])) for c in self.obstacle_clusters]
            if distances:
                d_min = min(distances)

        self.blend_alpha = self.calculate_blend_ratio(d_min)

        self.update_movement(dt)

        if self.check_collision(None):
            self.vx = 0.0
            self.vy = 0.0
            self.velocity = 0.0
            self.info_text.set_text("! COLLISION ! Speed Reset")
            self.goal = None
            self.goal_marker.set_data([], [])
            print("Collision detected! Vehicle stopped.")
        else:
            info_str = f"Speed: {self.velocity:.1f} m/s (vx:{self.vx:.1f}, vy:{self.vy:.1f})\n" \
                       f"Actual Frame dt: {self._current_frame_dt*1000:.1f} ms\n" \
                       f"Manual Direction: ({self.control_vector[0]:.1f}, {self.control_vector[1]:.1f})\n" \
                       f"Nearest Obstacle (tracked): {d_min:.2f} m, MPC Blend Ratio (Avoidance): {self.blend_alpha:.2f}"

            if self.goal:
                car_pos_np = np.array([self.x, self.y])
                goal_pos_np = np.array(self.goal)
                dist_to_goal = np.linalg.norm(car_pos_np - goal_pos_np)
                info_str += f"\nGoal: ({self.goal[0]:.1f}, {self.goal[1]:.1f}) Distance: {dist_to_goal:.1f} m"
                if dist_to_goal < self.car_radius + 0.5:
                    self.vx = 0.0
                    self.vy = 0.0
                    self.velocity = 0.0
                    self.goal = None
                    self.goal_marker.set_data([], [])
                    info_str += "\nGoal Reached!"
                    print("Goal reached! Vehicle stopped.")

            self.info_text.set_text(info_str)

        self.car_circle.center = (self.x, self.y)

        if hasattr(self, 'scan_points_for_plot') and isinstance(self.scan_points_for_plot, np.ndarray) and self.scan_points_for_plot.ndim == 2 and self.scan_points_for_plot.shape[1] == 2:
            self.scan_scatter.set_offsets(self.scan_points_for_plot)
        else:
            self.scan_scatter.set_offsets(np.empty((0, 2)))

        if self.trajectory:
            traj_np = np.array(self.trajectory)
            self.trajectory_line.set_data(traj_np[:, 0], traj_np[:, 1])
        else:
            self.trajectory_line.set_data([], [])

        if hasattr(self, 'mpc_trajectory') and self.mpc_trajectory:
            if len(self.mpc_trajectory) > 0:
                mpc_traj_np = np.array(self.mpc_trajectory)
                self.mpc_trajectory_line.set_data(mpc_traj_np[:, 0], mpc_traj_np[:, 1])
            else:
                self.mpc_trajectory_line.set_data([], [])
        else:
            self.mpc_trajectory_line.set_data([], [])

    def show(self):
        plt.show()

if __name__ == "__main__":
    controller = CarController()
    controller.show()
