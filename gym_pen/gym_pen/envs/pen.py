from PIL import Image
import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt

class PenEnv(gym.Env):
    def __init__(self, shape=(100, 100)):
        self.shape = shape
        # 환경 초기화 및 설정
        self.pen_position = np.zeros([2], dtype=np.intc)
        self.canvas = np.zeros(self.shape, dtype=np.float32)
        self.pen_mode = False  # True: pen down, False: pen up

        # 관찰 공간과 행동 공간 정의
        low = np.full(shape=(self.shape[0] * self.shape[1],), fill_value=0, dtype=np.float32)
        high = np.full(shape=(self.shape[0] * self.shape[1],), fill_value=1, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=np.max(self.shape), shape=(2,), dtype=np.float32)

    def reset(self, **kwargs):
        # 환경 리셋
        self.pen_position = np.zeros([2], dtype=np.intc)
        self.canvas = np.zeros(self.shape, dtype=np.float32)
        self.pen_mode = False
        return self.canvas.flatten(), {}

    def step(self, action):
        # 한 스텝 진행
        pre_pen_position = np.copy(self.pen_position)
        x, y, self.pen_mode = action
        self.pen_position = [x, y]
        self.pen_position[0] = np.clip(self.pen_position[0], 0, self.shape[0] - 1)
        self.pen_position[1] = np.clip(self.pen_position[1], 0, self.shape[1] - 1)

        if self.pen_mode:
            self.draw_line(pre_pen_position, self.pen_position)
        
        
        observation = self.canvas.flatten()
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def draw_line(self, pre_point, next_point):
        x0, y0 = pre_point
        x1, y1 = next_point
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx, sy = 1 if x0 < x1 else -1, 1 if y0 < y1 else -1
        err = dx - dy
        while (x0, y0) != (x1, y1):
            self.canvas[x0, y0] = 1
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    def initialize_with_noise(self, epsilon):
        # 노이즈로 초기화
        self.canvas = np.random.rand(self.shape) * epsilon

    def render(self):
        # 캔버스 렌더링
        render_arr = (self.canvas * 255).astype(np.uint8)
        render_arr = np.rot90(render_arr, 1)
        plt.imshow(render_arr, cmap='gray')
        plt.show()

    def close(self):
        # 렌더링 창 닫기
        plt.close()

    def save(self, file_path = '../image/'):
        save_array = (self.canvas * 255).astype(np.uint8)
        save_array = np.rot90(save_array, 1)
        image = Image.fromarray(save_array)
        image.save(file_path)