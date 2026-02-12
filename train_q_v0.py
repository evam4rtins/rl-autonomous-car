import matplotlib.pyplot as plt
from env.car_env_v0 import CarTrackEnvV0
from rl.qlearning import q_learning, evaluate_policy
from rl.discretization import create_discretizer


def main():

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
        psi_min=-3.1416 / 4,
        psi_max=3.1416 / 4,
        n_bins_y=n_bins_y,
        n_bins_psi=n_bins_psi,
    )

    # --------------------
    # Train Q-learning
    # --------------------
    Q, rewards = q_learning(
        env=env,
        discretize_state=discretize_state,
        state_shape=(n_bins_y, n_bins_psi),
        n_actions=env.action_space.n,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.05,
        n_episodes=500,
    )

    # --------------------
    # Plot learning curve
    # --------------------
    plt.figure()
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Q-learning Training (V0)")
    plt.show()

    # --------------------
    # Evaluate learned policy
    # --------------------
    print("Evaluating learned policy...")
    final_reward = evaluate_policy(env, Q, discretize_state, render=True)
    print("Final evaluation reward:", final_reward)

    env.close()


if __name__ == "__main__":
    main()
