import matplotlib.pyplot as plt
from env.car_env_v1 import CarTrackEnvV1
from rl.qlearning import q_learning
from rl.discretization import create_discretizer
import numpy as np
import pandas as pd
import os

os.makedirs("figures/discount", exist_ok=True)

env = CarTrackEnvV1(c1=0.3, c2=0.9)

n_bins_y = 15
n_bins_psi = 15

discretize_state = create_discretizer(
    y_min=-2.0, y_max=2.0,
    psi_min=-np.pi/4, psi_max=np.pi/4,
    n_bins_y=n_bins_y, n_bins_psi=n_bins_psi
)

# Parameters
discount_factors = [0.7, 0.85, 0.95, 0.99]
alpha = 0.1  # best learning rate from previous study
n_episodes = 800  

results = []

for gamma in discount_factors:
    print(f"Training with gamma = {gamma}")
    
    Q, rewards, max_q_changes = q_learning(
        env=env,
        discretize_state=discretize_state,
        state_shape=(n_bins_y, n_bins_psi),
        n_actions=env.action_space.n,
        alpha=alpha,
        gamma=gamma,
        n_episodes=n_episodes
    )
    
    # Compute metrics
    final_reward = np.mean(rewards[-50:])
    reward_std = np.std(rewards[-50:])
    avg_q_change = np.mean(max_q_changes[-50:])
    
    # Find convergence episode (first time reaching 90% of final performance)
    convergence_threshold = 0.9 * final_reward
    convergence_episodes = np.where(rewards >= convergence_threshold)[0]
    convergence_episode = convergence_episodes[0] if len(convergence_episodes) > 0 else n_episodes
    
    results.append({
        'gamma': gamma,
        'final_reward': final_reward,
        'reward_std': reward_std,
        'avg_q_change': avg_q_change,
        'convergence_episode': convergence_episode
    })
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(rewards)
    ax1.axhline(y=final_reward, color='g', linestyle='--', label=f'Final: {final_reward:.2f}')
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.set_title(f"Reward vs Episode (γ={gamma})")
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(max_q_changes, color='red')
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Max Q Change")
    ax2.set_title("Q-value Change")
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"figures/discount/gamma_{gamma}.png")
    plt.close()

df = pd.DataFrame(results)
df.to_csv("figures/discount_results.csv", index=False)
print("\nDiscount Factor Results:")
print(df.to_string())