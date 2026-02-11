from env.car_env_v0 import CarTrackEnvV0
import matplotlib.pyplot as plt

plt.ion()  # interactive mode

env = CarTrackEnvV0()
state, info = env.reset()

terminated = False
truncated = False

while not (terminated or truncated):
    action = env.action_space.sample()
    state, reward, terminated, truncated, info = env.step(action)
    env.render()

# Keep the window open at the end
plt.ioff()
plt.show()
env.close()
