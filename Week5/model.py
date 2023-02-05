import tensorflow as tf
from tensorflow.python.keras.activations import relu, softmax, sigmoid, tanh, elu, selu, softplus, softsign, leaky_relu

nl = {
    "tanh": tanh,
    "relu": relu,
    "sigmoid": sigmoid,
    "elu": elu,
    "selu": selu,
    "softplus": softplus,
    "softsign": softsign,
    "leaky_relu": leaky_relu,
    "none": lambda x: x,
}


class MyModel(tf.keras.Model):
    def __init__(self, name, filters, kernel_size, strides, pool_size, dropout, non_linearities):
        super().__init__(name=name)
        self.nl = nl[non_linearities]
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.Sequential = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters, kernel_size, strides, padding="same", activation=self.nl),
            tf.keras.layers.MaxPool2D(pool_size=pool_size),
            tf.keras.layers.Conv2D(filters, kernel_size, strides, padding="same", activation=self.nl),
            tf.keras.layers.MaxPool2D(pool_size=pool_size),
            tf.keras.layers.Conv2D(filters, kernel_size, strides, padding="same", activation=self.nl),
            tf.keras.layers.MaxPool2D(pool_size=pool_size),
            tf.keras.layers.Conv2D(filters, kernel_size, strides, padding="same", activation=self.nl),
            tf.keras.layers.MaxPool2D(pool_size=pool_size),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation=self.nl),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(8, activation=softmax)
        ])


if __name__ == "__main__":
    model = MyModel("MyModel", 32, 3, 1, 2, 0.5, "relu", 8)
    model.build(input_shape=(None, 224, 224, 3))
    model.summary()