import gym
import gym_pen
import numpy as np

shape = (500, 500)

env = gym.make('pen-v0')

observation, _ = env.reset()

import tensorflow as tf

def custom_model(shape=(100, 100)):
    # Define the layers
    array_input = tf.keras.layers.Input(shape=(shape[0], shape[1], 1))
    char_input = tf.keras.layers.Input(shape=(1,))
    index_input = tf.keras.layers.Input(shape=(1,))
    xy_input = tf.keras.layers.Input(shape=(2,))

    array_x = tf.keras.layers.Conv2D(15, 3, activation='relu')(array_input)
    array_x = tf.keras.layers.MaxPooling2D(5)(array_x)
    array_x = tf.keras.layers.Conv2D(15, 3, activation='relu')(array_x)
    array_x = tf.keras.layers.Flatten()(array_x)

    combined_x = tf.keras.layers.Concatenate()([array_x, index_input, xy_input])
    x = tf.keras.layers.Dense(64, activation='relu')(combined_x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)

    output_xy = tf.keras.layers.Dense(2, activation='relu')(x)

    model = tf.keras.Model(inputs=[array_input, index_input, xy_input], outputs=[output_xy])
    return model

model = custom_model(shape=shape)
model.summary()
env.initialize_with_noise(0.001)
# 모델 생성
model = custom_model()

# 모델 초기화
model.build(input_shape=[(None, 500, 500, 1), (None, 1), (None, 2)])

# 가상의 입력 데이터 생성 (with batch dimension)
array_input_data = np.expand_dims(env.model_canvas, axis=0)  # (1, 100, 100, 1)
index_input_data = np.array([[env.step_idx]])  # (1, 1)
xy_input_data = np.array([env.model_pen_position])  # (1, 2)


# 모델에 입력 데이터 전달하여 출력 얻기
output_xy = model.predict([array_input_data, index_input_data, xy_input_data])

np.shape(output_xy[0])
print(output_xy[0,0])