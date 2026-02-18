import matplotlib.pyplot as plt
from env.car_env_v1 import CarTrackEnvV1
from rl.qlearning import q_learning
from rl.discretization import create_discretizer
import numpy as np
import pandas as pd
import psutil
import os

os.makedirs("figures/discretization", exist_ok=True)

env = CarTrackEnvV1(c1=0.3, c2=0.9)  # Medium complexity

# Test different resolutions
resolutions = [10, 20, 40]
alpha = 0.1
gamma = 0.99
n_episodes = 500

results = []

for n_bins in resolutions:
    print(f"\nTesting {n_bins}x{n_bins} grid")
    
    discretize_state = create_discretizer(
        y_min=-2.0, y_max=2.0,
        psi_min=-np.pi/4, psi_max=np.pi/4,
        n_bins_y=n_bins, n_bins_psi=n_bins
    )
    
    # Memory usage
    mem_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    Q, rewards, max_q_changes = q_learning(
        env=env,
        discretize_state=discretize_state,
        state_shape=(n_bins, n_bins),
        n_actions=env.action_space.n,
        alpha=alpha,
        gamma=gamma,
        n_episodes=n_episodes
    )
    
    mem_after = psutil.Process().memory_info().rss / 1024 / 1024
    memory_mb = mem_after - mem_before
    
    # Metrics
    final_reward = np.mean(rewards[-50:])
    reward_std = np.std(rewards[-50:])
    
    # Convergence speed (episodes to reach 90% of final)
    convergence_idx = np.argmax(np.array(rewards) >= 0.9 * final_reward)
    convergence_episodes = convergence_idx + 1 if convergence_idx > 0 else n_episodes
    
    results.append({
        'resolution': f"{n_bins}x{n_bins}",
        'n_states': n_bins * n_bins,
        'n_q_values': n_bins * n_bins * 3,
        'memory_mb': memory_mb,
        'final_reward': final_reward,
        'reward_std': reward_std,
        'convergence_episodes': convergence_episodes
    })
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label=f'{n_bins}x{n_bins}')
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(f"Learning Curves for Different Resolutions")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"figures/discretization/res_{n_bins}.png")
    plt.close()

# Comparison plot
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
res = [r['resolution'] for r in results]
final_rew = [r['final_reward'] for r in results]
plt.bar(res, final_rew)
plt.title("Final Reward")
plt.ylabel("Reward")

plt.subplot(1, 3, 2)
conv = [r['convergence_episodes'] for r in results]
plt.bar(res, conv)
plt.title("Episodes to Converge")
plt.ylabel("Episodes")

plt.subplot(1, 3, 3)
mem = [r['memory_mb'] for r in results]
plt.bar(res, mem)
plt.title("Memory Usage")
plt.ylabel("MB")

plt.tight_layout()
plt.savefig("figures/discretization/comparison.png")
plt.close()

df = pd.DataFrame(results)
df.to_csv("figures/discretization_results.csv", index=False)
print("\nDiscretization Results:")
print(df.to_string())