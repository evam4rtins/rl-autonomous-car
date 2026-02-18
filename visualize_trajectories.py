import matplotlib.pyplot as plt
from env.car_env_v1 import CarTrackEnvV1
from rl.qlearning import q_learning
from rl.discretization import create_discretizer
import numpy as np
import pandas as pd
import os

plt.ioff()
os.makedirs("figures/trajectories", exist_ok=True)

def collect_trajectory(env, Q, discretize_state, max_steps=500):
    """Run a greedy policy and collect the trajectory"""
    state, _ = env.reset()
    trajectory = []
    actions_taken = []
    
    terminated = False
    truncated = False
    step = 0

    # Collect trajectory until episode ends or max steps reached
    while not (terminated or truncated) and step < max_steps:
        trajectory.append(env.pos.copy())
        
        state_idx = discretize_state(state)
        action = np.argmax(Q[state_idx])
        actions_taken.append(action)
        
        state, reward, terminated, truncated, _ = env.step(action)
        step += 1
    
    return np.array(trajectory), actions_taken

def plot_trajectory_comparison(env_configs, results, title="Trajectory Comparison"):
    """Plot trajectories from different policies on the same track"""
    
    fig, axes = plt.subplots(1, len(env_configs), figsize=(15, 5))
    if len(env_configs) == 1:
        axes = [axes]
    
    for idx, (env_name, config) in enumerate(env_configs.items()):
        ax = axes[idx]
        
        env = config['env_class'](**config['env_params'])
        
        # Plot track
        ax.plot(env.track_x, env.track_y, 'k-', lw=2, label='Track Center', alpha=0.7)
        
        # Plot track boundaries
        ax.fill_between(env.track_x, env.track_y - 1.0, env.track_y + 1.0, 
                        color='gray', alpha=0.2, label='Track Bounds')
        
        # Plot trajectories for different alphas
        colors = plt.cm.viridis(np.linspace(0, 1, len(results[env_name])))
        
        for (alpha, data), color in zip(results[env_name].items(), colors):
            trajectory = data['trajectory']
            ax.plot(trajectory[:, 0], trajectory[:, 1], 
                   color=color, lw=1.5, label=f'α={alpha}', alpha=0.7)
            
            # Mark start and end
            ax.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=8)
            ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'ro', markersize=8)
        
        ax.plot([], [], 'go', markersize=8, label='Start')
        ax.plot([], [], 'ro', markersize=8, label='End')
        
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_title(f"{env_name} Complexity\nMean Curvature: {config['mean_curvature']:.3f}")
        ax.set_aspect('equal')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig("figures/trajectories/comparison_all.png", dpi=150, bbox_inches='tight')
    plt.close(fig)  

def plot_trajectory_with_actions(env, Q, discretize_state, alpha, complexity_name, save_dir="figures/trajectories"):
    """Plot trajectory colored by action taken"""
    
    trajectory, actions = collect_trajectory(env, Q, discretize_state)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    action_colors = ['red', 'green', 'blue']
    action_labels = ['Left', 'Straight', 'Right']
    
    for action in range(3):
        mask = np.array(actions) == action
        if np.any(mask):
            ax1.scatter(trajectory[mask, 0], trajectory[mask, 1], 
                       c=action_colors[action], label=action_labels[action], 
                       s=10, alpha=0.6)
    
    # Plot track
    ax1.plot(env.track_x, env.track_y, 'k-', lw=2, label='Track Center', alpha=0.5)
    ax1.set_xlabel("X Position")
    ax1.set_ylabel("Y Position")
    ax1.set_title(f"Trajectory Colored by Action (α={alpha}) - {complexity_name}")
    ax1.legend()
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Lateral error over time
    lateral_errors = []
    for pos in trajectory:
        # Calculate lateral error using the environment's method
        distances = np.linalg.norm(env.track - pos, axis=1)
        idx = np.argmin(distances)
        track_point = env.track[idx]
        
        # Get track heading at that point
        if idx < len(env.track) - 1:
            next_point = env.track[idx + 1]
            track_heading = np.arctan2(next_point[1] - track_point[1],
                                      next_point[0] - track_point[0])
        else:
            track_heading = 0
        
        # Signed lateral error
        normal = np.array([-np.sin(track_heading), np.cos(track_heading)])
        e_y = np.dot(pos - track_point, normal)
        lateral_errors.append(e_y)
    
    ax2.plot(lateral_errors, 'b-', lw=1.5)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='Center')
    ax2.fill_between(range(len(lateral_errors)), -1, 1, color='gray', alpha=0.2, label='Track Bounds (±1)')
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Lateral Error")
    ax2.set_title("Lateral Error Over Trajectory")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f"Policy Analysis: α={alpha}, {complexity_name} Complexity", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/trajectory_alpha_{alpha}_{complexity_name.lower()}.png", dpi=150, bbox_inches='tight')
    plt.close(fig)  
    
    return trajectory, actions

def plot_best_policies_comparison(env_configs, best_policies, save_dir="figures/trajectories"):
    """Plot the best policy for each complexity level"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (complexity_name, config) in enumerate(env_configs.items()):
        ax = axes[idx]
        
        env = config['env_class'](**config['env_params'])
        
        alpha = best_policies[complexity_name]['alpha']
        Q = best_policies[complexity_name]['Q']
        
        trajectory, actions = collect_trajectory(env, Q, discretize_state)
        
        # Plot track
        ax.plot(env.track_x, env.track_y, 'k-', lw=2, label='Track', alpha=0.7)
        
        # Plot track boundaries
        ax.fill_between(env.track_x, env.track_y - 1.0, env.track_y + 1.0, 
                        color='gray', alpha=0.2, label='Track Bounds')
        
        # Plot trajectory
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'r-', lw=2, label='Agent Path')
        
        # Mark start and end
        ax.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=10, label='Start')
        ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'ro', markersize=10, label='End')
        
        # Calculate performance metrics
        final_reward = best_policies[complexity_name]['final_reward']
        mean_curv = config['mean_curvature']
        
        ax.set_title(f"{complexity_name} Complexity\nα={alpha}, Mean Curv: {mean_curv:.3f}\nFinal Reward: {final_reward:.1f}")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_aspect('equal')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle("Best Policies Across Complexity Levels", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/best_policies_comparison.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

def main():
    complexity_levels = {
        'Low': {'c1': 0.3, 'c2': 0.0},
        'Medium': {'c1': 0.3, 'c2': 0.9},
        'High': {'c1': 0.5, 'c2': 1.5}
    }
    
    global discretize_state  
    n_bins_y = 15
    n_bins_psi = 15
    discretize_state = create_discretizer(
        y_min=-2.0, y_max=2.0,
        psi_min=-np.pi/4, psi_max=np.pi/4,
        n_bins_y=n_bins_y, n_bins_psi=n_bins_psi
    )
    
    alphas_to_test = [0.01, 0.05, 0.1]
    
    all_results = {}
    env_configs = {}
    best_policies = {}
    
    for complexity_name, params in complexity_levels.items():
        print(f"\n=== Testing {complexity_name} Complexity ===")
        
        env = CarTrackEnvV1(**params)
        mean_curvature = np.mean(env.curvature)
        print(f"Mean Curvature: {mean_curvature:.4f}")
        
        all_results[complexity_name] = {}
        env_configs[complexity_name] = {
            'env_class': CarTrackEnvV1,
            'env_params': params,
            'mean_curvature': mean_curvature
        }
        
        best_reward = -np.inf
        best_alpha = None
        best_Q = None
        
        for alpha in alphas_to_test:
            print(f"  Training with α={alpha}...")
            
            # Train agent
            Q, rewards, _ = q_learning(
                env=env,
                discretize_state=discretize_state,
                state_shape=(n_bins_y, n_bins_psi),
                n_actions=env.action_space.n,
                alpha=alpha,
                gamma=0.99,
                n_episodes=500
            )
            
            final_reward = np.mean(rewards[-50:])
            
            trajectory, actions = collect_trajectory(env, Q, discretize_state)
            
            all_results[complexity_name][alpha] = {
                'trajectory': trajectory,
                'actions': actions,
                'final_reward': final_reward,
                'Q': Q
            }
            
            if final_reward > best_reward:
                best_reward = final_reward
                best_alpha = alpha
                best_Q = Q
            
            # Plot individual trajectory 
            plot_trajectory_with_actions(env, Q, discretize_state, alpha, complexity_name)
        
        best_policies[complexity_name] = {
            'alpha': best_alpha,
            'final_reward': best_reward,
            'Q': best_Q
        }
        print(f"  Best policy: α={best_alpha} with reward={best_reward:.2f}")
    
    print("\n=== Generating comparison plots ===")
    plot_trajectory_comparison(env_configs, all_results, 
                              "Trajectory Comparison Across Complexity Levels")
    
    plot_best_policies_comparison(env_configs, best_policies)
    
    summary_data = []
    for complexity_name, results in all_results.items():
        for alpha, data in results.items():
            action_dist = np.bincount(data['actions'], minlength=3)
            summary_data.append({
                'Complexity': complexity_name,
                'Alpha': alpha,
                'Final Reward': f"{data['final_reward']:.2f}",
                'Trajectory Length': len(data['trajectory']),
                'Left %': f"{action_dist[0]/len(data['actions'])*100:.1f}%",
                'Straight %': f"{action_dist[1]/len(data['actions'])*100:.1f}%",
                'Right %': f"{action_dist[2]/len(data['actions'])*100:.1f}%"
            })
    
    df = pd.DataFrame(summary_data)
    print("\n=== Summary Table ===")
    print(df.to_string(index=False))
    df.to_csv("figures/trajectories/trajectory_summary.csv", index=False)
    
    best_df = pd.DataFrame([
        {'Complexity': k, 'Best Alpha': v['alpha'], 'Best Reward': f"{v['final_reward']:.2f}"}
        for k, v in best_policies.items()
    ])
    best_df.to_csv("figures/trajectories/best_policies.csv", index=False)
    
    print("\nAll figures saved to figures/trajectories/")

if __name__ == "__main__":
    main()