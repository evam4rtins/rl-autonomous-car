import numpy as np

def q_learning(
    env, discretize_state, state_shape, n_actions,
    alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995,
    epsilon_min=0.05, n_episodes=500
):
    Q = np.zeros((*state_shape, n_actions))
    episode_rewards = []
    max_q_changes = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        state_idx = discretize_state(state)
        total_reward = 0
        terminated = False
        truncated = False
        Q_prev = Q.copy()

        while not (terminated or truncated):
            # Epsilon-greedy
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state_idx])

            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state_idx = discretize_state(next_state)

            best_next_action = np.argmax(Q[next_state_idx])
            td_target = reward + gamma * Q[next_state_idx][best_next_action]
            td_error = td_target - Q[state_idx][action]

            Q[state_idx][action] += alpha * td_error

            state_idx = next_state_idx
            total_reward += reward

        # Record metrics
        episode_rewards.append(total_reward)
        max_q_changes.append(np.max(np.abs(Q - Q_prev)))

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return Q, episode_rewards, max_q_changes

def evaluate_policy(env, Q, discretize_state, render=True):
    """
    Run one greedy episode using the learned Q-table.
    """

    state, _ = env.reset()
    terminated = False
    truncated = False
    total_reward = 0

    while not (terminated or truncated):

        state_idx = discretize_state(state)
        action = np.argmax(Q[state_idx])  # greedy action

        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        if render:
            env.render()

    return total_reward
