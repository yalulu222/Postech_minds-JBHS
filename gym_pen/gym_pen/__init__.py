from gym.envs.registration import register

register(
    id='pen-v0',
    entry_point='gym_pen.envs:PenEnv',
)
