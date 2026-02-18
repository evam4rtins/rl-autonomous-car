import matplotlib.pyplot as plt
from env.car_env_v1 import CarTrackEnvV1
from rl.qlearning import q_learning
from rl.discretization import create_discretizer
import numpy as np
import pandas as pd
import os

os.makedirs("figures/complexity", exist_ok=True)

complexity_levels = [
    {'name': 'Low', 'c1': 0.3, 'c2': 0.0},
    {'name': 'Medium', 'c1': 0.3, 'c2': 0.9},
    {'name': 'High', 'c1': 0.5, 'c2': 1.5}
]

# Discretization
n_bins_y = 15
n_bins_psi = 15

discretize_state = create_discretizer(
    y_min=-2.0, y_max=2.0,
    psi_min=-np.pi/4, psi_max=np.pi/4,
    n_bins_y=n_bins_y, n_bins_psi=n_bins_psi
)

# Test learning rates
learning_rates = [0.01, 0.05, 0.1, 0.3, 0.5]
n_episodes = 500
results = []

for level in complexity_levels:
    print(f"\n=== Testing {level['name']} Complexity ===")
    env = CarTrackEnvV1(c1=level['c1'], c2=level['c2'])
    
    # Print curvature metrics
    metrics = env.get_curvature_metrics()
    print(f"Mean Curvature: {metrics['mean_curvature']:.4f}")
    print(f"Max Curvature: {metrics['max_curvature']:.4f}")
    
    for alpha in learning_rates:
        print(f"  Training with alpha = {alpha}")
        
        Q, rewards, max_q_changes = q_learning(
            env=env,
            discretize_state=discretize_state,
            state_shape=(n_bins_y, n_bins_psi),
            n_actions=env.action_space.n,
            alpha=alpha,
            gamma=0.99,
            n_episodes=n_episodes
        )
        
        results.append({
            'complexity': level['name'],
            'c1': level['c1'],
            'c2': level['c2'],
            'mean_curvature': metrics['mean_curvature'],
            'alpha': alpha,
            'final_reward': np.mean(rewards[-50:]),
            'reward_std': np.std(rewards[-50:]),
            'avg_q_change': np.mean(max_q_changes[-50:]),
            'convergence_episode': np.argmax(np.array(rewards) >= 0.9 * np.max(rewards)) + 1
        })
        
        # Plot learning curves
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        ax1.plot(rewards)
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Total Reward")
        ax1.set_title(f"{level['name']} Complexity: Reward (α={alpha})")
        ax1.grid(True)
        
        ax2.plot(max_q_changes, color='red')
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Max Q Change")
        ax2.set_title("Q-value Change")
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"figures/complexity/{level['name']}_alpha_{alpha}.png")
        plt.close()

df = pd.DataFrame(results)
df.to_csv("figures/complexity_results.csv", index=False)
print("\nResults saved to figures/complexity_results.csv")