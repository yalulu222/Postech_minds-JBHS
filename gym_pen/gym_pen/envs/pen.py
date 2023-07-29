from PIL import Image
import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine

class PenEnv(gym.Env):
    def __init__(self, shape=(100, 100)):
        self.shape = shape
        # 환경 초기화 및 설정
        self.model_pen_position = np.zeros([2], dtype=np.intc)
        self.data_pen_position = np.zeros([2], dtype=np.intc)

        self.model_canvas = np.zeros(self.shape, dtype=np.float32)
        self.data_canvas = np.zeros(self.shape, dtype=np.float32)

        self.model_pen_mode = False  # True: pen down, False: pen up
        self.data_pen_mode = False  # True: pen down, False: pen up

        self.pen_trace = np.zeros([100,3], dtype=np.intc)
        self.step_idx = 0

        # 관찰 공간과 행동 공간 정의
        low = np.full(shape=(self.shape[0] * self.shape[1],), fill_value=0, dtype=np.float32)
        high = np.full(shape=(self.shape[0] * self.shape[1],), fill_value=1, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=np.max(self.shape), shape=(2,), dtype=np.float32)

    def reset(self, **kwargs):
        # 환경 리셋
        self.model_pen_position = np.zeros([2], dtype=np.intc)
        self.data_pen_position = np.zeros([2], dtype=np.intc)

        self.model_canvas = np.zeros(self.shape, dtype=np.float32)
        self.data_canvas = np.zeros(self.shape, dtype=np.float32)

        self.model_pen_mode = False  # True: pen down, False: pen up
        self.data_pen_mode = False  # True: pen down, False: pen up

        self.pen_trace = np.zeros([100,3], dtype=np.intc)
        self.step_idx = 0

        return self.model_canvas.flatten(), {}

    def step(self, model_action, data_action):
        # 한 스텝 진행
        model_pre_pen_position = np.copy(self.model_pen_position)
        model_x, model_y, self.model_pen_mode = model_action
        self.pen_trace[self.step_idx] = [model_x, model_y, self.model_pen_position]
        self.step_idx += 1
        
        self.model_pen_position = [model_x, model_y]
        self.model_pen_position[0] = np.clip(self.model_pen_position[0], 0, self.shape[0] - 1)
        self.model_pen_position[1] = np.clip(self.model_pen_position[1], 0, self.shape[1] - 1)

        if self.model_pen_mode:
            self.draw_line(model_pre_pen_position, self.model_pen_position, 'model')
        
        data_pre_pen_position = np.copy(self.data_pen_position)
        data_x, data_y, self.data_pen_mode = data_action
        
        self.data_pen_position = [data_x, data_y]
        self.data_pen_position[0] = np.clip(self.data_pen_position[0], 0, self.shape[0] - 1)
        self.data_pen_position[1] = np.clip(self.data_pen_position[1], 0, self.shape[1] - 1)

        if self.data_pen_mode:
            self.draw_line(data_pre_pen_position, self.data_pen_position, 'data')

        # Flatten the 2D arrays into 1D vectors
        A_flat = self.model_canvas.flatten()
        B_flat = self.data_canvas.flatten()
        # Calculate cosine similarity
        cosine_similarity = 1 - cosine(A_flat, B_flat)
        
        observation = self.model_canvas.flatten()
        reward = cosine_similarity
        if self.step_idx == 99:
            terminated = True
        else:
            terminated = False
        truncated = False
        info = {}

        return observation, reward, terminated, truncated, info

    def draw_line(self, pre_point, next_point, canvas_type):
        x0, y0 = pre_point
        x1, y1 = next_point
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx, sy = 1 if x0 < x1 else -1, 1 if y0 < y1 else -1
        err = dx - dy
        while (x0, y0) != (x1, y1):
            if canvas_type == 'model':
                self.model_canvas[x0, y0] = 1
            elif canvas_type == 'data':
                self.data_canvas[x0, y0] = 1
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    def initialize_with_noise(self, epsilon):
        # Unpack the shape tuple
        height, width = self.shape

        # 노이즈로 초기화
        self.model_canvas = np.random.rand(height, width) * epsilon
        self.data_canvas = np.random.rand(height, width) * epsilon


    def render(self, canvas_type):
        if canvas_type == 'model':
            render_arr = (self.model_canvas * 255).astype(np.uint8)
        elif canvas_type == 'data':
            render_arr = (self.data_canvas * 255).astype(np.uint8)
        # 캔버스 렌더링
        render_arr = np.rot90(render_arr, 1)
        plt.imshow(render_arr, cmap='gray')
        plt.show()

    def close(self):
        # 렌더링 창 닫기
        plt.close()

    def save(self, canvas_type, file_path):
        if canvas_type == 'model':
            save_array = (self.model_canvas * 255).astype(np.uint8)
        elif canvas_type == 'data':
            save_array = (self.data_canvas * 255).astype(np.uint8)
        save_array = np.rot90(save_array, 1)
        image = Image.fromarray(save_array)
        image.save(file_path)