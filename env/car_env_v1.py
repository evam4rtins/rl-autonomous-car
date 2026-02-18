import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt


class CarTrackEnvV1(gym.Env):
    """
    Version 1:
    - Parameterized track family: y(x) = sin(c1*x) + 0.5*sin(c2*x)
    - Curvature metric calculation
    - State: (lateral error, heading error)
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, c1=0.3, c2=0.0, max_steps=500):
        super().__init__()
        self.fig = None
        self.ax = None
        
        # Track parameters
        self.c1 = c1
        self.c2 = c2
        
        self.max_steps = max_steps
        self.current_step = 0
        
        # Create parameterized track
        self.track_x = np.linspace(0, 20, 200)
        self.track_y = np.sin(self.c1 * self.track_x) + 0.5 * np.sin(self.c2 * self.track_x)
        self.track = np.vstack([self.track_x, self.track_y]).T
        
        # Compute curvature along track
        self.curvature = self._compute_curvature()
        
        # Actions: 0 = left, 1 = straight, 2 = right
        self.action_space = spaces.Discrete(3)
        
        # State = [lateral error, heading error]
        self.observation_space = spaces.Box(
            low=np.array([-2.0, -np.pi/4], dtype=np.float32),
            high=np.array([2.0, np.pi/4], dtype=np.float32),
            dtype=np.float32
        )
    
    def _compute_curvature(self):
        """Compute curvature κ(x) = |y''| / (1 + (y')^2)^(3/2)"""
        dx = self.track_x[1] - self.track_x[0]
        
        # First derivative (central difference)
        y_prime = np.gradient(self.track_y, dx)
        
        # Second derivative
        y_double_prime = np.gradient(y_prime, dx)
        
        # Curvature formula
        curvature = np.abs(y_double_prime) / (1 + y_prime**2)**(1.5)
        
        return curvature
    
    def get_curvature_metrics(self):
        """Return statistics about track curvature"""
        return {
            'mean_curvature': np.mean(self.curvature),
            'max_curvature': np.max(self.curvature),
            'std_curvature': np.std(self.curvature)
        }
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.pos = np.array([0.0, 0.0], dtype=np.float32)
        self.heading = 0.0
        self.speed = 0.2
        
        self.current_step = 0
        
        state = self._get_state()
        info = {'curvature_metrics': self.get_curvature_metrics()}
        
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
        
        info = {'curvature_at_position': self.curvature[np.argmin(np.abs(self.track_x - self.pos[0]))]}
        
        return state, reward, terminated, truncated, info
    
    def render(self):
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(10, 6))
            plt.ion()
            self.ax.plot(self.track_x, self.track_y, 'k-', lw=2, label='Track')
            self.ax.set_xlim(-1, 21)
            self.ax.set_ylim(-3, 3)
            self.ax.set_aspect('equal')
            self.ax.set_title(f"Car Tracking: c1={self.c1}, c2={self.c2}, "
                            f"Mean Curvature={np.mean(self.curvature):.3f}")
            self.car_patch, = self.ax.plot([], [], 'ro', markersize=8)
            self.heading_arrow = None
            self.trajectory_x = []
            self.trajectory_y = []
        
        # Update car position
        self.car_patch.set_data([self.pos[0]], [self.pos[1]])
        
        # Update trajectory
        self.trajectory_x.append(self.pos[0])
        self.trajectory_y.append(self.pos[1])
        if len(self.trajectory_x) > 1:
            self.ax.plot(self.trajectory_x[-2:], self.trajectory_y[-2:], 'r--', alpha=0.5)
        
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