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
    if name == "baseline_8_corrected":
    # Model BO (JOHNNY)
        Sequential = tf.keras.Sequential([
            # First convolutional block --> input (256, 256, 3) --> output (128, 128, 32)
            tf.keras.layers.Conv2D(filters[0], kernel_size[1], strides, padding="same", activation=nl,
                                input_shape=(256, 256, 3)),
            tf.keras.layers.MaxPool2D(pool_size=pool_size),
            tf.keras.layers.Dropout(dropout_rate),
            # Second convolutional block  --> input (128, 128, 32) --> output (64, 64, 64)
            tf.keras.layers.Conv2D(filters[1], kernel_size[1], strides, padding="same", activation=nl),
            tf.keras.layers.MaxPool2D(pool_size=pool_size),
            tf.keras.layers.Dropout(dropout_rate),
            # Third convolutional block  --> input (64, 64, 64) --> output (32, 32, 128)
            tf.keras.layers.Conv2D(filters[2], kernel_size[1], strides, padding="same", activation=nl),
            tf.keras.layers.Conv2D(filters[2], kernel_size[1], strides, padding="same", activation=nl),
            tf.keras.layers.MaxPool2D(pool_size=pool_size),
            tf.keras.layers.Dropout(dropout_rate),
            # Batch normalization layer --> input (32, 32, 128) --> output (32, 32, 128)
            # Flatten and feed to output layer
            tf.keras.layers.GlobalAveragePooling2D(),
            # Feed the network to the fully connected layers
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            # Classification layer
            tf.keras.layers.Dense(8, activation='softmax')
        ])

    elif name == "Olorente":
        Sequential = tf.keras.Sequential([
            # First convolutional block --> 
            tf.keras.layers.Conv2D(filters[0], kernel_size[0], strides, activation=nl,
                                input_shape=(32, 32, 3)),
            tf.keras.layers.MaxPool2D(pool_size=pool_size),
            # Second convolutional block  --> 
            tf.keras.layers.Conv2D(filters[0], kernel_size[0], strides, activation=nl),
            tf.keras.layers.MaxPool2D(pool_size=pool_size),
            # Flatten and feed to output layer
            tf.keras.layers.Flatten(),
            # Classification layer
            tf.keras.layers.Dense(8, activation='softmax')
        ])

    elif name == "baseline_8_lowcost_1":
        Sequential = tf.keras.Sequential([  # use 5x5 kernel size, and remove 3rd and 1024 dense layer.
            # First convolutional block --> input (256, 256, 3) --> output (128, 128, 32)
            tf.keras.layers.Conv2D(filters[0], kernel_size[0], strides, padding="same", activation=nl,
                                input_shape=(256, 256, 3)),
            tf.keras.layers.MaxPool2D(pool_size=pool_size),
            # Second convolutional block  --> input (128, 128, 32) --> output (64, 64, 64)
            tf.keras.layers.Conv2D(filters[1], kernel_size[0], strides, padding="same", activation=nl),
            tf.keras.layers.MaxPool2D(pool_size=pool_size),
            # Third convolutional block  --> input (64, 64, 64) --> output (32, 32, 128)
            tf.keras.layers.Conv2D(filters[2], kernel_size[0], strides, padding="same", activation=nl),
            tf.keras.layers.MaxPool2D(pool_size=pool_size),
            # Batch normalization layer --> input (32, 32, 128) --> output (32, 32, 128)
            tf.keras.layers.BatchNormalization(),
            # Flatten and feed to output layer
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(dropout_rate),
            # Feed the network to the fully connected layers
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            # Classification layer
            tf.keras.layers.Dense(8, activation='softmax')
        ])

    elif name == "medium_extended":  # val acc 0.59.. sense data augmentation
        Sequential = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), padding="same", input_shape=(64, 64, 3), activation='relu'),
            # tf.keras.layers.BatchNormalization(axis=1),
            tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),
            # tf.keras.layers.Dropout(0.25),

            tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation='relu'),
            # tf.keras.layers.BatchNormalization(axis=1),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            # tf.keras.layers.Dropout(0.25),

            tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation='relu'),
            # tf.keras.layers.BatchNormalization(axis=1),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            # tf.keras.layers.Dropout(0.25),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512),
            tf.keras.layers.Dense(64, activation='relu'),
            # tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            # tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(8,activation="softmax")
        ])

    elif name == "basic":
        Sequential = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), padding="same", input_shape=(64, 64, 3), activation='relu'),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            #tf.keras.layers.Flatten(),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(8, activation='softmax')
        ])
    
    elif name == "basic2":
        Sequential = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), padding="same", input_shape=(64, 64, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            #tf.keras.layers.Flatten(),
            tf.keras.layers.GlobalAveragePooling2D(),
            #tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(8, activation='softmax')
        ])
    elif name == "basic3":
        Sequential = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), padding="same", input_shape=(64, 64, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            #tf.keras.layers.Flatten(),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(8, activation='softmax')
        ])
    elif name == "basic4":
        Sequential = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), padding="same", input_shape=(64, 64, 3), activation='relu'),
            tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation='relu'),
            tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation='relu'),
            tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512),
            #tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(8,activation="softmax")
        ])
    elif name == "basic5":
        Sequential = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), padding="same", input_shape=(64, 64, 3), activation='relu'),
            tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation='relu'),
            tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(8,activation="softmax")
        ])
    elif name == "medium_256input_2blocks":
        Sequential = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), padding="same", input_shape=(64, 64, 3), activation='relu'),
            tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation='relu'),
            tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation='relu'),
            tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation='relu'),
            
            tf.keras.layers.BatchNormalization(axis=1),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),

            # tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation='relu'),
            # tf.keras.layers.BatchNormalization(axis=1),
            # tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            # tf.keras.layers.Dropout(0.25),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(8,activation="softmax")
        ])
    return Sequential


def AutoencoderModel(name, filters, kernel_size, strides, pool_size, dropout_rate, non_linearities):
    pass  # TODO


# if __name__ == "__main__":
#     model = MyModel("MyModel", 32, 3, 1, 2, 0.5, "relu", 8)
#     model.build(input_shape=(None, 256, 256, 3))
#     model.summary()
