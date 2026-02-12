import matplotlib.pyplot as plt
from env.car_env_v0 import CarTrackEnvV0
from rl.qlearning import q_learning
from rl.discretization import create_discretizer
import numpy as np
import os  

os.makedirs("figures", exist_ok=True)

# --------------------
# Environment
# --------------------
env = CarTrackEnvV0()

# --------------------
# Discretization
# --------------------
n_bins_y = 15
n_bins_psi = 15

discretize_state = create_discretizer(
    y_min=-2.0,
    y_max=2.0,
    psi_min=-np.pi/4,
    psi_max=np.pi/4,
    n_bins_y=n_bins_y,
    n_bins_psi=n_bins_psi,
)

# --------------------
# Learning rate analysis
# --------------------
learning_rates = [0.01, 0.05, 0.1, 0.3, 0.5, 0.8, 1]
n_episodes = 1000

def compute_metrics(rewards, max_q_changes, last_n=50):
    final_reward = np.mean(rewards[-last_n:])
    plateau = final_reward
    threshold = 0.9 * plateau
    episodes_to_90 = np.argmax(np.array(rewards) >= threshold) + 1
    avg_max_q = np.mean(max_q_changes[-last_n:])
    reward_std = np.std(rewards[-last_n:])
    return final_reward, episodes_to_90, avg_max_q, reward_std

metrics_table = []

for alpha in learning_rates:
    print(f"Training with alpha = {alpha}")

    Q, rewards, max_q_changes = q_learning(
        env=env,
        discretize_state=discretize_state,
        state_shape=(n_bins_y, n_bins_psi),
        n_actions=env.action_space.n,
        alpha=alpha,
        gamma=0.99,
        n_episodes=n_episodes
    )

    # Compute metrics
    final_reward, episodes_to_90, avg_max_q, reward_std = compute_metrics(rewards, max_q_changes)
    metrics_table.append([alpha, final_reward, episodes_to_90, avg_max_q, reward_std])

    # --------------------
    # Plot reward per episode
    # --------------------
    plt.figure(figsize=(8,4))
    plt.plot(rewards, color='blue')
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(f"Q-learning: Reward vs Episode (α={alpha})")
    plt.grid(True)
    plt.savefig(f"figures/reward_alpha_{alpha}.png") 
    plt.close()

    # --------------------
    # Plot max Q-value change per episode
    # --------------------
    plt.figure(figsize=(8,4))
    plt.plot(max_q_changes, color='red')
    plt.xlabel("Episode")
    plt.ylabel("Max Q-value Change")
    plt.title(f"Q-learning: Q-value Change vs Episode (α={alpha})")
    plt.grid(True)
    plt.savefig(f"figures/qchange_alpha_{alpha}.png")
    plt.close()

import pandas as pd
df_metrics = pd.DataFrame(metrics_table, columns=["α", "Final Reward", "Episodes to 90%", "Avg Max Q", "Reward Std"])
print(df_metrics)
df_metrics.to_csv("figures/v0_metrics.csv", index=False)
