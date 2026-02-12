import matplotlib.pyplot as plt
from env.car_env_v0 import CarTrackEnvV0


def main():

    plt.ion() 

    env = CarTrackEnvV0()
    state, info = env.reset()

    terminated = False
    truncated = False
    total_reward = 0

    while not (terminated or truncated):

        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        env.render()

    plt.ioff()
    plt.show()

    print("Total reward (random policy):", total_reward)

    env.close()


if __name__ == "__main__":
    main()
