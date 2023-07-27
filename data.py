import numpy as np
import gym
import gym_pen
import os

def read_data(file_path, file_name):
    xy_array = []
    xy_coord = [0,0]
    with open(os.path.join(file_path, file_name), 'r') as file:
        pen_mode = 0
        is_start_box = False
        for line in file:
            if line == '.PEN_DOWN\n':
                is_start_box = True
                pen_mode = 1

            if is_start_box:
                if line == '.PEN_UP\n' or line == '.PEN_DOWN\n' or line == '.START_BOX\n':
                    pen_mode = 0
                else:
                    try:
                        xy_coord = line.replace('\n','').replace('\t',' ').split(' ')
                        xy_coord = [int(item) for item in xy_coord if item != '']
                        xy_coord.append(pen_mode)
                        xy_array.append(xy_coord)
                        pen_mode = 1
                    except:
                        pass
    return np.array(xy_array).astype(np.intc)

def scale_data(xy_array, shape):
    min_x = np.min(xy_array[:, 0])
    max_x = np.max(xy_array[:, 0])
    min_y = np.min(xy_array[:, 1])
    max_y = np.max(xy_array[:, 1])

    xy_array_scaled = np.copy(xy_array)
    xy_array_scaled[:, 0] = (shape[0] - 1) * (xy_array[:, 0] - min_x) / (max_x - min_x)
    xy_array_scaled[:, 1] = (shape[1] - 1) * (xy_array[:, 1] - min_y) / (max_y - min_y)

    return xy_array_scaled


file_path = 'unipen/CDROM/train_r01_v07/include/apa/data/apa00/'
# file_lst = os.listdir(file_path)

file_name = 'app0032.dat'

xy_array = read_data('unipen/CDROM/train_r01_v07/include/apa/data/apa00/','app0032.dat') 

min_x = np.min(xy_array[:, 0])
max_x = np.max(xy_array[:, 0])
min_y = np.min(xy_array[:, 1])
max_y = np.max(xy_array[:, 1])

shape = (max_x - min_x, max_y - min_y)

scaled_xy_array = scale_data(xy_array, shape)

# PenEnv 환경을 생성합니다. 원하는 shape를 인자로 넘겨서 환경을 초기화합니다.
env = gym.make('pen-v0', shape=shape)

# 환경 초기화
observation, _ = env.reset()

# 스케일된 좌표를 이용하여 환경 실행
for i in range(len(scaled_xy_array)):
    env.step(scaled_xy_array[i])

directory = 'images/'
if not os.path.exists(directory):
    os.makedirs(directory)

file_path = f'images/{file_name[:-4]}.jpg'
env.save(file_path)
# file_path = 'unipen/CDROM/train_r01_v07/include/aga/data/'
# file_lst = os.listdir(file_path)

    
# for file_name in file_lst:
#     xy_array = read_data(file_path, file_name) 

#     min_x = np.min(xy_array[:, 0])
#     max_x = np.max(xy_array[:, 0])
#     min_y = np.min(xy_array[:, 1])
#     max_y = np.max(xy_array[:, 1])

#     shape = (max_x - min_x, max_y - min_y)

#     scaled_xy_array = scale_data(xy_array, shape)

#     # PenEnv 환경을 생성합니다. 원하는 shape를 인자로 넘겨서 환경을 초기화합니다.
#     env = gym.make('pen-v0', shape=shape)

#     # 환경 초기화
#     observation, _ = env.reset()

#     # 스케일된 좌표를 이용하여 환경 실행
#     for i in range(len(scaled_xy_array)):
#         env.step(scaled_xy_array[i])

#     directory = 'images/'
#     if not os.path.exists(directory):
#         os.makedirs(directory)

#     file_path = f'images/{file_name[:-4]}.jpg'
#     env.save(file_path)