import numpy as np
import gym
import gym_pen
import os

def read_data(data_file):
    pen_mode = 0
    index = -1
    all_index_array = []
    index_array = []
    for line in data_file:
        if line[0] == '.':
            if line == '.PEN_DOWN\n':
                if index != -1:
                    all_index_array.append(index_array)
                index_array = []
                index += 1
                pen_mode = 0
            elif line == '.PEN_UP\n':
                pen_mode = 0
            else:
                pass
        elif line == '\n':
            pass
        else:
            try:            
                index_array.append(line.split())
                pen_mode = 1
            except:
                pass
    
    return all_index_array

def scale_data(xy_array, shape):
    min_x = np.min(xy_array[:, 0])
    max_x = np.max(xy_array[:, 0])
    min_y = np.min(xy_array[:, 1])
    max_y = np.max(xy_array[:, 1])

    xy_array_scaled = np.copy(xy_array)
    xy_array_scaled[:, 0] = (shape[0] - 1) * (xy_array[:, 0] - min_x) / (max_x - min_x)
    xy_array_scaled[:, 1] = (shape[1] - 1) * (xy_array[:, 1] - min_y) / (max_y - min_y)

    return xy_array_scaled

def bezier_curve(control_points, num_points=100):
    n = len(control_points) - 1
    t = np.linspace(0, 1, num_points)
    
    def binomial_coefficient(n, k):
        return np.math.factorial(n) / (np.math.factorial(k) * np.math.factorial(n - k))
    
    def bernstein_poly(i, n, t):
        return binomial_coefficient(n, i) * (t ** i) * ((1 - t) ** (n - i))
    
    curve_points = np.zeros((num_points, 2))
    for i in range(num_points):
        curve_points[i, 0] = sum(bernstein_poly(j, n, t[i]) * control_points[j][0] for j in range(n + 1))
        curve_points[i, 1] = sum(bernstein_poly(j, n, t[i]) * control_points[j][1] for j in range(n + 1))
    
    return curve_points

def add_third_value(data):
    result_data = []
    n = len(data)
    
    for i in range(n):
        if i == 0 or i == n - 1:
            result_data.append([data[i][0], data[i][1], 0])
        else:
            result_data.append([data[i][0], data[i][1], 1])
    
    return result_data
import pandas as pd


file_list = os.listdir('unipen/data')
for file_name in file_list:
  file_name = file_name[:-4]
  try:
    with open(os.path.join(f'unipen/include/{file_name}.dat'), 'r') as file:
      data_array = read_data(file)

    df = pd.read_csv(f'unipen/data/{file_name}.csv')

    df[['start_index', 'end_index']] = df['index'].str.split('-', expand=True)
    df['start_index'] = df['start_index'].fillna(df['start_index']).astype(int)
    df['end_index'] = df['end_index'].fillna(df['start_index']).astype(int)


    df = df.drop('index', axis=1)

    for i in range(len(df)):
      end_index, start_index, label = df['end_index'][i], df['start_index'][i], df['label'][i]

      if end_index == start_index:
        xy_array = data_array[start_index]

        xy_array = np.array(xy_array, dtype= np.intc)
        min_x = np.min(xy_array[:, 0])
        max_x = np.max(xy_array[:, 0])
        min_y = np.min(xy_array[:, 1])
        max_y = np.max(xy_array[:, 1])
        shape = (max_x - min_x, max_y - min_y)
        xy_array = scale_data(xy_array, shape)
        xy_array = bezier_curve(xy_array).astype(np.intc)
        xy_array = add_third_value(xy_array)
      else:
        xy_array_1 = data_array[start_index]
        xy_array_1 = np.array(xy_array_1, dtype= np.intc)

        xy_array_2 = data_array[end_index]
        xy_array_2 = np.array(xy_array_2, dtype= np.intc)
        
        min_x = np.min([ np.min(xy_array_1[:, 0]) , np.min(xy_array_2[:, 0])])
        max_x = np.max([ np.max(xy_array_1[:, 0]) , np.max(xy_array_2[:, 0])])
        min_y = np.min([ np.min(xy_array_1[:, 1]) , np.min(xy_array_2[:, 1])])
        max_y = np.max([ np.max(xy_array_1[:, 1]) , np.max(xy_array_2[:, 1])])


        shape = (max_x - min_x, max_y - min_y)
        xy_array_1 = scale_data(xy_array_1, shape)
        xy_array_2 = scale_data(xy_array_2, shape)
        xy_array_1 = bezier_curve(xy_array_1, 50).astype(np.intc)
        xy_array_2 = bezier_curve(xy_array_2, 50).astype(np.intc)
        xy_array = np.concatenate((add_third_value(xy_array_1), add_third_value(xy_array_2)), axis=0)
      directory = f'data/{label}/'
      if not os.path.exists(directory):
        os.makedirs(directory)
      pd.DataFrame(xy_array, columns=['x', 'y', 'pen_mode']).to_csv(f'{directory}{file_name}_{i}.csv', index=False)
  except:
    print(file_name)