import numpy as np
import matplotlib.pyplot as plt
from env.car_env_v0 import CarTrackEnvV0

env = CarTrackEnvV0()

# Number of bins
n_bins_y = 15
n_bins_psi = 15

# State bounds (must match env observation space)
y_min, y_max = -2.0, 2.0
psi_min, psi_max = -np.pi/4, np.pi/4

# Create bin edges
y_bins = np.linspace(y_min, y_max, n_bins_y)
psi_bins = np.linspace(psi_min, psi_max, n_bins_psi)

def discretize_state(state):
    """
    Discretize the continuous state into bin indices.
    """
    e_y, e_psi = state

    y_idx = np.digitize(e_y, y_bins) - 1
    psi_idx = np.digitize(e_psi, psi_bins) - 1

    # Clip to ensure valid index
    y_idx = np.clip(y_idx, 0, n_bins_y - 1)
    psi_idx = np.clip(psi_idx, 0, n_bins_psi - 1)

    return y_idx, psi_idx

# Init Q-table
n_actions = env.action_space.n

Q = np.zeros((n_bins_y, n_bins_psi, n_actions))

# Hyperparameters to study numerical analysis (change them and see what happens)
alpha = 0.1        # learning rate
gamma = 0.99       # discount factor
epsilon = 1.0      # exploration
epsilon_decay = 0.995
epsilon_min = 0.05

n_episodes = 500

# Training
episode_rewards = []

for episode in range(n_episodes):

    state, _ = env.reset()
    y_idx, psi_idx = discretize_state(state)

    total_reward = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):

        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[y_idx, psi_idx])

        next_state, reward, terminated, truncated, _ = env.step(action)
        next_y_idx, next_psi_idx = discretize_state(next_state)

        # Q-learning update
        best_next_action = np.argmax(Q[next_y_idx, next_psi_idx])

        td_target = reward + gamma * Q[next_y_idx, next_psi_idx, best_next_action]
        td_error = td_target - Q[y_idx, psi_idx, action]

        Q[y_idx, psi_idx, action] += alpha * td_error

        y_idx, psi_idx = next_y_idx, next_psi_idx
        total_reward += reward

    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    episode_rewards.append(total_reward)

    if episode % 50 == 0:
        print(f"Episode {episode}, Reward: {total_reward:.2f}")

# plot learning curve
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Q-learning Training")
plt.show()

# Playback trained policy with trajectory
import matplotlib.pyplot as plt
plt.ion()

state, _ = env.reset()
terminated = False
truncated = False

while not (terminated or truncated):
    y_idx, psi_idx = discretize_state(state)
    action = np.argmax(Q[y_idx, psi_idx])  # best action from Q-table
    state, reward, terminated, truncated, _ = env.step(action)
    env.render()  # show car + trajectory

plt.ioff()
plt.show()
env.close()
