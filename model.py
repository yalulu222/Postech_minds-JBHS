import tensorflow as tf 

class CustomModel(tf.keras.Model):
    def __init__(self, shape=(100,100)):
        super(CustomModel, self).__init__()
        self.shape = shape
        # Define the layers in the constructor
        self.array_input_layer = tf.keras.layers.Input(shape=(shape, 1))
        self.char_input_layer = tf.keras.layers.Input(shape=(1,))
        self.index_input_layer = tf.keras.layers.Input(shape=(1,))
        self.xy_input_layer = tf.keras.layers.Input(shape=(2,))

        self.array_x = tf.keras.layers.Conv2D(15, 3, activation='relu')
        self.max_pooling = tf.keras.layers.MaxPooling2D(5)
        self.flatten = tf.keras.layers.Flatten()
        self.concat = tf.keras.layers.Concatenate()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.output_xy = tf.keras.layers.Dense(2, activation='linear')
        self.output_penmode = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        # Define the forward pass in the call method
        array_input, char_input = inputs

        array_x = self.array_x(array_input)
        array_x = self.max_pooling(array_x)
        array_x = self.array_x(array_x)
        array_x = self.flatten(array_x)

        combined_x = self.concat([char_input, array_x, self.index_input_layer, self.xy_input_layer])
        x = self.dense1(combined_x)
        x = self.dense2(x)

        output_xy = self.output_xy(x)
        output_penmode = self.output_penmode(x)

        return output_xy, output_penmode

# Create an instance of the custom model
model = CustomModel()

# Compile the model and define the optimizer, loss functions, etc.
model.compile(optimizer='adam', loss=['mse', 'binary_crossentropy'])

# Print the model summary
model.summary()