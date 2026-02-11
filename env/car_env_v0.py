import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt


class CarTrackEnvV0(gym.Env):
    """
    Version 0:
    - Curved track
    - State = (lateral error e_y, heading error e_psi)
    - Action = steering only
    - Constant speed
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, max_steps=500):
        super().__init__()
        self.fig = None
        self.ax = None


        self.max_steps = max_steps
        self.current_step = 0

        # Create a curved track using waypoints
        self.track_x = np.linspace(0, 20, 200)
        self.track_y = np.sin(0.3 * self.track_x)
        self.track = np.vstack([self.track_x, self.track_y]).T

        # Actions: 0 = left, 1 = straight, 2 = right
        self.action_space = spaces.Discrete(3)

        # State = [lateral error, heading error]
        self.observation_space = spaces.Box(
            low=np.array([-2.0, -np.pi/4], dtype=np.float32),
            high=np.array([2.0, np.pi/4], dtype=np.float32),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.pos = np.array([0.0, 0.0], dtype=np.float32)
        self.heading = 0.0
        self.speed = 0.2

        self.current_step = 0

        state = self._get_state()
        info = {}

        return state, info

    def _get_state(self):
        distances = np.linalg.norm(self.track - self.pos, axis=1)
        idx = np.argmin(distances)

        track_point = self.track[idx]

        if idx < len(self.track) - 1:
            next_point = self.track[idx + 1]
        else:
            next_point = track_point

        track_heading = np.arctan2(
            next_point[1] - track_point[1],
            next_point[0] - track_point[0]
        )

        # Lateral error (signed)
        normal = np.array([
            -np.sin(track_heading),
             np.cos(track_heading)
        ])

        e_y = np.dot(self.pos - track_point, normal)

        # Heading error
        e_psi = self.heading - track_heading

        return np.array([e_y, e_psi], dtype=np.float32)

    def step(self, action):
        self.current_step += 1

        # Steering update
        if action == 0:
            self.heading -= 0.05
        elif action == 2:
            self.heading += 0.05

        # Move forward
        self.pos[0] += self.speed * np.cos(self.heading)
        self.pos[1] += self.speed * np.sin(self.heading)

        state = self._get_state()
        e_y, e_psi = state

        # Reward: stay centered and aligned
        reward = - (e_y**2 + e_psi**2)

        terminated = False
        truncated = False

        # Car leaves track
        if abs(e_y) > 2.0:
            terminated = True
            reward -= 5

        # Reaches end of track
        if self.pos[0] >= self.track_x[-1]:
            terminated = True
            reward += 10

        # Max steps reached
        if self.current_step >= self.max_steps:
            truncated = True

        info = {}

        return state, reward, terminated, truncated, info

    def render(self):
        if self.fig is None:
            self.fig, self.ax = plt.subplots()
            plt.ion()  # interactive mode
            self.ax.plot(self.track_x, self.track_y, 'k-', lw=2, label='Track')
            self.ax.set_xlim(-1, 21)
            self.ax.set_ylim(-3, 3)
            self.ax.set_aspect('equal')
            self.ax.set_title("Car Tracking Environment")
            self.car_patch, = self.ax.plot([], [], 'ro', markersize=8)
            self.heading_arrow = None
            self.trajectory_x = []
            self.trajectory_y = []

        # Update car position
        self.car_patch.set_data([self.pos[0]], [self.pos[1]])

        # Update trajectory
        self.trajectory_x.append(self.pos[0])
        self.trajectory_y.append(self.pos[1])
        self.ax.plot(self.trajectory_x, self.trajectory_y, 'r--', alpha=0.5)

        # Remove old heading arrow
        if self.heading_arrow:
            self.heading_arrow.remove()

        # Draw heading arrow
        arrow_length = 0.5
        dx = arrow_length * np.cos(self.heading)
        dy = arrow_length * np.sin(self.heading)
        self.heading_arrow = self.ax.arrow(
            self.pos[0], self.pos[1], dx, dy,
            head_width=0.1, head_length=0.1, fc='b', ec='b'
        )

        plt.pause(0.01)

