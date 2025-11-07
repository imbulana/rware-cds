import gymnasium as gym

env = gym.make("rware:rware-tiny-2ag-v2")
obs = env.reset()

while True:
    actions = env.action_space.sample()
    print(actions)
    n_obs, reward, done, truncated, info = env.step(actions)
    print(done)
    print(reward)

    env.render()
# env.close()