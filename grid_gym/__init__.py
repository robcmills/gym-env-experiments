from gymnasium.envs.registration import register

register(
    id="grid_gym/GridWorld-v0",
    entry_point="grid_gym.envs:GridWorldEnv",
)
