import tensorflow as tf
from tensorflow.python.keras.activations import relu, softmax, sigmoid, tanh, elu, selu, softplus, softsign, leaky_relu

_nl = {
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

"""
class MyModel(tf.keras.Model):
    def __init__(self, name, filters, kernel_size, strides, pool_size, dropout, non_linearities):
        super().__init__(name=name)
        self.nl = nl[non_linearities]
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
    def call(self, inputs):
        return self.Sequential(inputs)
"""


def MyModel(name, filters, kernel_size, strides, pool_size, dropout_rate, non_linearities):
    nl = _nl[non_linearities]
    # Sequential = tf.keras.Sequential([
    #     tf.keras.layers.Conv2D(filters, kernel_size, strides, padding="same", activation=nl, input_shape=(224, 224, 3)),
    #     tf.keras.layers.MaxPool2D(pool_size=pool_size),
    #     tf.keras.layers.Conv2D(filters, kernel_size, strides, padding="same", activation=nl),
    #     tf.keras.layers.MaxPool2D(pool_size=pool_size),
    #     tf.keras.layers.Conv2D(filters, kernel_size, strides, padding="same", activation=nl),
    #     tf.keras.layers.MaxPool2D(pool_size=pool_size),
    #     tf.keras.layers.Conv2D(filters, kernel_size, strides, padding="same", activation=nl),
    #     tf.keras.layers.MaxPool2D(pool_size=pool_size),
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(256, activation=nl),
    #     tf.keras.layers.Dropout(dropout_rate),
    #     tf.keras.layers.Dense(8, activation=softmax)
    # ])

    """ 0.72 val accuracy
    Sequential = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters[0], kernel_size[1], strides, padding="same", activation=nl, input_shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(filters[0], kernel_size[1], strides, padding="same", activation=nl),
        tf.keras.layers.MaxPool2D(pool_size=pool_size),
        tf.keras.layers.Conv2D(filters[1], kernel_size[1], strides, padding="same", activation=nl),
        tf.keras.layers.Conv2D(filters[1], kernel_size[1], strides, padding="same", activation=nl),
        tf.keras.layers.MaxPool2D(pool_size=pool_size),
        tf.keras.layers.BatchNormalization(),
       
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(64, activation='relu'),
        #tf.keras.layers.GlobalAveragePooling2D(),
        #tf.keras.layers.Dense(128, activation ='relu'),
        
        tf.keras.layers.Dense(8, activation ='softmax')
    ])
    """
    Sequential = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters[0], kernel_size[1], strides, padding="same", activation=nl,
                               input_shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(filters[0], kernel_size[1], strides, padding="same", activation=nl),
        tf.keras.layers.MaxPool2D(pool_size=pool_size),
        tf.keras.layers.Conv2D(filters[1], kernel_size[1], strides, padding="same", activation=nl),
        tf.keras.layers.Conv2D(filters[1], kernel_size[1], strides, padding="same", activation=nl),
        tf.keras.layers.MaxPool2D(pool_size=pool_size),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(256, activation='relu'),
        # tf.keras.layers.GlobalAveragePooling2D(),
        # tf.keras.layers.Dense(128, activation ='relu'),

        tf.keras.layers.Dense(8, activation='softmax')
    ])

    return Sequential


if __name__ == "__main__":
    model = MyModel("MyModel", 32, 3, 1, 2, 0.5, "relu", 8)
    model.build(input_shape=(None, 224, 224, 3))
    model.summary()
