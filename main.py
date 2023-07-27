import numpy as np
import gym
import gym_pen

shape = (1000, 480)

# PenEnv 환경을 생성합니다. 원하는 shape를 인자로 넘겨서 환경을 초기화합니다.
env = gym.make('pen-v0', shape=shape)

# 환경 초기화
observation, _ = env.reset()

