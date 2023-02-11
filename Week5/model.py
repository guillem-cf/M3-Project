
# from tensorflow.keras.activations import relu, softmax, sigmoid, tanh, elu, selu, softplus, softsign, leaky_relu
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
import tensorflow as tf


# _nl = {
#     "tanh": tanh,
#     "relu": relu,
#     "sigmoid": sigmoid,
#     "elu": elu,
#     "selu": selu,
#     "softplus": softplus,
#     "softsign": softsign,
#     "leaky_relu": leaky_relu,
#     "none": lambda x: x,
# }




def MyModel(name, 
            img_dim,
            num_blocks,
            second_layer,
            third_layer,
            num_denses,
            dim_dense,
            filters1,
            filters2, 
            batch_norm,
            dropout,
            dropout_range,
            kernel_size,
            strides, 
            non_linearities,
            initializer,
            pool_size):


    # def mish(x):
    #     return x * tf.math.tanh(tf.math.softplus(x))

    # # Define mish using keras backend
    def mish(x):
        return x * K.tanh(K.softplus(x))

    # get_custom_objects().update({'mish': Activation(mish)})

    print("Creating model: ", name)
    print("num_blocks: ", num_blocks)
    print("img_dim: ", img_dim)
    print("second_layer: ", second_layer)
    print("num_denses: ", num_denses)
    print("filters1: ", filters1)
    print("filters2: ", filters2)
    print("batch_norm: ", batch_norm)
    print("dropout: ", dropout)
    print("dropout_range: ", dropout_range)
    print("kernel_size: ", kernel_size)
    print("strides: ", strides)
    print("non_linearities: ", non_linearities)
    print("initializer: ", initializer)
    print("pool_size: ", pool_size)

    if non_linearities == 'mish':
        print("-----------------MISH-----------------")
        non_linearities = mish

    

    if name == "baseline_8_corrected":
        # Model BO (JOHNNY)
        Sequential = tf.keras.Sequential([
            # First convolutional block --> input (256, 256, 3) --> output (128, 128, 32)
            tf.keras.layers.Conv2D(64, 3, strides, padding="same", activation="leaky_relu",
                                   kernel_initializer="HeUniform",
                                   input_shape=(64, 64, 3)),
            tf.keras.layers.Conv2D(64, 3, strides, padding="same", activation="leaky_relu",
                                   kernel_initializer="HeUniform"),
            tf.keras.layers.MaxPool2D(pool_size=pool_size),
            tf.keras.layers.BatchNormalization(),
            # Second convolutional block  --> input (128, 128, 32) --> output (64, 64, 64)
            tf.keras.layers.Conv2D(64, 3, strides, padding="same", activation="leaky_relu",
                                   kernel_initializer="HeUniform"),
            tf.keras.layers.Conv2D(64, 3, strides, padding="same", activation="leaky_relu",
                                   kernel_initializer="HeUniform"),
            tf.keras.layers.MaxPool2D(pool_size=pool_size),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Conv2D(128, 3, strides, padding="same", activation="leaky_relu",
                                   kernel_initializer="HeUniform"),
            tf.keras.layers.Conv2D(128, 3, strides, padding="same", activation="leaky_relu",
                                   kernel_initializer="HeUniform"),
            tf.keras.layers.MaxPool2D(pool_size=pool_size),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Conv2D(256, 3, strides, padding="same", activation="leaky_relu",
                                   kernel_initializer="HeUniform"),
            tf.keras.layers.Conv2D(256, 3, strides, padding="same", activation="leaky_relu",
                                   kernel_initializer="HeUniform"),
            tf.keras.layers.MaxPool2D(pool_size=pool_size),
            tf.keras.layers.BatchNormalization(),

            # Batch normalization layer --> input (32, 32, 128) --> output (32, 32, 128)
            # Flatten and feed to output layer
            tf.keras.layers.GlobalAveragePooling2D(),
            # Feed the network to the fully connected layers
            tf.keras.layers.Dense(256, activation="leaky_relu", kernel_initializer="HeUniform"),
            tf.keras.layers.Dropout(0.6),
            tf.keras.layers.Dense(128, activation="leaky_relu", kernel_initializer="HeUniform"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(64, activation="leaky_relu", kernel_initializer="HeUniform"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(8, activation='softmax', kernel_initializer="HeUniform")
        ])

    elif name == "best_model_sweep":
        # Model BO (JOHNNY)
        Sequential = tf.keras.Sequential()
        # First convolutional block --> input (256, 256, 3) --> output (128, 128, 32)
        Sequential.add(tf.keras.layers.Conv2D(filters1, kernel_size, strides, padding="same", activation=non_linearities,
                                              kernel_initializer=initializer,
                                              input_shape=(img_dim, img_dim, 3)))
        if second_layer:
            Sequential.add(tf.keras.layers.Conv2D(filters1, kernel_size, strides, padding="same", activation=non_linearities,
                                                kernel_regularizer=regularizers.l2(1e-4),
                                                kernel_initializer=initializer))
        Sequential.add(tf.keras.layers.MaxPool2D(pool_size=pool_size))
        if batch_norm:
            Sequential.add(tf.keras.layers.BatchNormalization())
        
        if num_blocks > 1:
            # Second convolutional block  --> input (128, 128, 32) --> output (64, 64, 64)
            Sequential.add(tf.keras.layers.Conv2D(filters2, kernel_size, strides, padding="same", activation=non_linearities,
                                                   kernel_regularizer=regularizers.l2(1e-4),
                                                   kernel_initializer=initializer))
            if second_layer:
                Sequential.add(tf.keras.layers.Conv2D(filters2, kernel_size, strides, padding="same", activation=non_linearities,
                                                  kernel_regularizer=regularizers.l2(1e-4),
                                                  kernel_initializer=initializer))
            Sequential.add(tf.keras.layers.MaxPool2D(pool_size=pool_size))
            if batch_norm:
                Sequential.add(tf.keras.layers.BatchNormalization())

        if num_blocks > 2:
            # Third convolutional block 
            Sequential.add(tf.keras.layers.Conv2D(128, kernel_size, strides, padding="same", activation=non_linearities,
                                                    kernel_regularizer=regularizers.l2(1e-4),
                                                    kernel_initializer=initializer))
            if second_layer:
                Sequential.add(tf.keras.layers.Conv2D(128, kernel_size, strides, padding="same", activation=non_linearities,
                                                    kernel_regularizer=regularizers.l2(1e-4),
                                                    kernel_initializer=initializer))
            Sequential.add(tf.keras.layers.MaxPool2D(pool_size=pool_size))
            if batch_norm:
                Sequential.add(tf.keras.layers.BatchNormalization())

        if num_blocks > 3:
            # Fourth convolutional block
            Sequential.add(tf.keras.layers.Conv2D(256, kernel_size, strides, padding="same", activation=non_linearities,
                                                    kernel_regularizer=regularizers.l2(1e-4),
                                                    kernel_initializer=initializer))
            if second_layer:
                Sequential.add(tf.keras.layers.Conv2D(256, kernel_size, strides, padding="same", activation=non_linearities,
                                                    kernel_regularizer=regularizers.l2(1e-4),
                                                    kernel_initializer=initializer))
                if third_layer:
                    Sequential.add(tf.keras.layers.Conv2D(256, kernel_size, strides, padding="same", activation=non_linearities,
                                                    kernel_regularizer=regularizers.l2(1e-4),
                                                    kernel_initializer=initializer))
            Sequential.add(tf.keras.layers.MaxPool2D(pool_size=pool_size))
            if batch_norm:
                Sequential.add(tf.keras.layers.BatchNormalization())

        if num_blocks > 4:
            # Fourth convolutional block
            Sequential.add(tf.keras.layers.Conv2D(512, kernel_size, strides, padding="same", activation=non_linearities,
                                                    kernel_regularizer=regularizers.l2(1e-4),
                                                    kernel_initializer=initializer))
            if second_layer:
                Sequential.add(tf.keras.layers.Conv2D(512, kernel_size, strides, padding="same", activation=non_linearities,
                                                    kernel_regularizer=regularizers.l2(1e-4),
                                                    kernel_initializer=initializer))
            Sequential.add(tf.keras.layers.MaxPool2D(pool_size=pool_size))
            if batch_norm:
                Sequential.add(tf.keras.layers.BatchNormalization())


        Sequential.add(tf.keras.layers.GlobalAveragePooling2D())

        
        if num_denses == 4:
            Sequential.add(tf.keras.layers.Dense(dim_dense, activation=non_linearities, 
                            kernel_initializer=initializer, kernel_regularizer=regularizers.l2(1e-4)))
            if dropout:
                Sequential.add(tf.keras.layers.Dropout(dropout_range + 0.4))
            Sequential.add(tf.keras.layers.Dense(int(dim_dense //2), activation=non_linearities, 
                            kernel_initializer=initializer, kernel_regularizer=regularizers.l2(1e-4)))
            if dropout:
                Sequential.add(tf.keras.layers.Dropout(dropout_range + 0.3))
            Sequential.add(tf.keras.layers.Dense(int(dim_dense //4), activation=non_linearities, 
                            kernel_initializer=initializer, kernel_regularizer=regularizers.l2(1e-4)))
            if dropout:
                Sequential.add(tf.keras.layers.Dropout(dropout_range + 0.2))
            Sequential.add(tf.keras.layers.Dense(int(dim_dense //8), activation=non_linearities, 
                            kernel_initializer=initializer, kernel_regularizer=regularizers.l2(1e-4)))
            if dropout:
                Sequential.add(tf.keras.layers.Dropout(dropout_range))
        if num_denses == 3:
            Sequential.add(tf.keras.layers.Dense(dim_dense, activation=non_linearities, 
                            kernel_initializer=initializer, kernel_regularizer=regularizers.l2(1e-4)))
            if dropout:
                Sequential.add(tf.keras.layers.Dropout(dropout_range[0]))
            Sequential.add(tf.keras.layers.Dense(int(dim_dense //2), activation=non_linearities, 
                            kernel_initializer=initializer, kernel_regularizer=regularizers.l2(1e-4)))
            if dropout:
                Sequential.add(tf.keras.layers.Dropout(dropout_range[1]))
            Sequential.add(tf.keras.layers.Dense(int(dim_dense //8), activation=non_linearities, 
                            kernel_initializer=initializer, kernel_regularizer=regularizers.l2(1e-4)))
            if dropout:
                Sequential.add(tf.keras.layers.Dropout(dropout_range[2]))
        if num_denses == 2:
            Sequential.add(tf.keras.layers.Dense(128, activation=non_linearities, 
                            kernel_initializer=initializer, kernel_regularizer=regularizers.l2(1e-4)))
            if dropout:
                Sequential.add(tf.keras.layers.Dropout(dropout_range + 0.3))
            Sequential.add(tf.keras.layers.Dense(64, activation=non_linearities, 
                            kernel_initializer=initializer, kernel_regularizer=regularizers.l2(1e-4)))
            if dropout:
                Sequential.add(tf.keras.layers.Dropout(dropout_range))
        if num_denses == 1:
            Sequential.add(tf.keras.layers.Dense(64, activation=non_linearities, 
                            kernel_initializer=initializer, kernel_regularizer=regularizers.l2(1e-4)))
            if dropout:
                Sequential.add(tf.keras.layers.Dropout(dropout_range))

        Sequential.add(tf.keras.layers.Dense(8, activation='softmax', kernel_initializer=initializer))


    elif name == "withoutdense_sweep":
        def conv2d(filters, kernel_size, padding='same', strides=1):
            def f(x):
                x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                strides=strides, padding=padding,
                                kernel_regularizer=regularizers.l2(1e-4))(x)
                x = tf.keras.layers.Activation(non_linearities)(x)
                x = tf.keras.layers.BatchNormalization()(x)
                return x

            return f
        # Model BO (JOHNNY)
        # First convolutional block --> input (256, 256, 3) --> output (128, 128, 32)
        inputs = tf.keras.layers.Input(shape=(img_dim, img_dim, 3))

        initial_filters = 32
        repetitions = num_blocks

        x = conv2d(initial_filters, kernel_size=3)(inputs)
        x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)

        filters = initial_filters
        for r in repetitions:
            for i in range(r):
                x = conv2d(filters, kernel_size=3)(x)
            x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
            filters *= 2

        x = tf.keras.layers.Dropout(dropout_range)(x)
        x = conv2d(8, kernel_size=1, padding='valid')(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Activation('softmax')(x)

        # x = tf.keras.layers.Dense(8, activation='softmax', kernel_initializer=initializer)(x)

        Sequential = tf.keras.Model(inputs=inputs, outputs=x)
    
    elif name == "residual_connections1":
        input = tf.keras.Input(shape=(64, 64, 3))
        x = tf.keras.layers.Conv2D(filters1, (3, 3), padding="same", activation=non_linearities)(input)
        x = tf.keras.layers.Conv2D(filters1, (3, 3), padding="same", activation=non_linearities)(x)
        # x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        b1_out = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(filters1, (3, 3), padding="same", activation=non_linearities)(b1_out)
        x = tf.keras.layers.Conv2D(filters1, (3, 3), padding="same", activation=non_linearities)(x)
        # x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        b2_out = tf.keras.layers.BatchNormalization()(x)
        res_1 = tf.keras.layers.Add()([b1_out, b2_out])
        x = tf.keras.layers.Conv2D(filters1, (3, 3), padding="same", activation=non_linearities)(res_1)
        x = tf.keras.layers.Conv2D(filters1, (3, 3), padding="same", activation=non_linearities)(x)
        # x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        b3_out = tf.keras.layers.BatchNormalization()(x)
        res_2 = tf.keras.layers.Add()([res_1, b3_out])
        x = tf.keras.layers.GlobalAveragePooling2D()(res_2)
        # Feed the network to the fully connected layers
        x = tf.keras.layers.Dense(256, activation=non_linearities, kernel_initializer=initializer)(x)
        x = tf.keras.layers.Dropout(0.6)(x)
        x = tf.keras.layers.Dense(128, activation=non_linearities, kernel_initializer=initializer)(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(64, activation=non_linearities, kernel_initializer=initializer)(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        output = tf.keras.layers.Dense(8, activation='softmax', kernel_initializer=initializer)(x)

        Sequential = tf.keras.Model(inputs=input, outputs=output)

    elif name == "residual_connections2":
        input = tf.keras.Input(shape=(64, 64, 3))
        x = tf.keras.layers.Conv2D(filters1, (3, 3), padding="same", activation=non_linearities)(input)
        x = tf.keras.layers.Conv2D(filters1, (3, 3), padding="same", activation=non_linearities)(x)
        # x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        b1_out = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(filters1, (3, 3), padding="same", activation=non_linearities)(b1_out)
        x = tf.keras.layers.Conv2D(filters1, (3, 3), padding="same", activation=non_linearities)(x)
        # x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        b2_out = tf.keras.layers.BatchNormalization()(x)
        res_1 = tf.keras.layers.Add()([b1_out, b2_out])
        x = tf.keras.layers.Conv2D(filters1, (3, 3), padding="same", activation=non_linearities)(res_1)
        x = tf.keras.layers.Conv2D(filters1, (3, 3), padding="same", activation=non_linearities)(x)
        # x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        b3_out = tf.keras.layers.BatchNormalization()(x)
        res_2 = tf.keras.layers.Add()([b1_out, b2_out, b3_out])
        x = tf.keras.layers.GlobalAveragePooling2D()(res_2)
        # Feed the network to the fully connected layers
        x = tf.keras.layers.Dense(256, activation=non_linearities, kernel_initializer=initializer)(x)
        x = tf.keras.layers.Dropout(0.6)(x)
        x = tf.keras.layers.Dense(128, activation=non_linearities, kernel_initializer=initializer)(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(64, activation=non_linearities, kernel_initializer=initializer)(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        output = tf.keras.layers.Dense(8, activation='softmax', kernel_initializer=initializer)(x)

        Sequential = tf.keras.Model(inputs=input, outputs=output)



    elif name == "medium_extended":  # val acc 0.59.. sense data augmentation
        Sequential = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), padding="same", input_shape=(64, 64, 3), activation='relu'),
            #  tf.keras.layers.BatchNormalization(axis=1),
            tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),
            # tf.keras.layers.Dropout(0.25),

            tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation='relu'),
            #  tf.keras.layers.BatchNormalization(axis=1),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            #  tf.keras.layers.Dropout(0.25),

            tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation='relu'),
            #  tf.keras.layers.BatchNormalization(axis=1),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            #  tf.keras.layers.Dropout(0.25),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512),
            tf.keras.layers.Dense(64, activation='relu'),
            #  tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            #  tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(8, activation="softmax")
        ])

    elif name == "basic":
        Sequential = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), padding="same", input_shape=(64, 64, 3), activation='relu'),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            # tf.keras.layers.Flatten(),
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
            # tf.keras.layers.Flatten(),
            tf.keras.layers.GlobalAveragePooling2D(),
            # tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(8, activation='softmax')
        ])
    elif name == "basic3":
        Sequential = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), padding="same", input_shape=(64, 64, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            # tf.keras.layers.Flatten(),
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
            # tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(8, activation="softmax")
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
            tf.keras.layers.Dense(8, activation="softmax")
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
            tf.keras.layers.Dense(8, activation="softmax")
        ])
    return Sequential


def AutoencoderModel(name, filters, kernel_size, strides, pool_size, dropout_rate, non_linearities):
    pass  # TODO

# if __name__ == "__main__":
#     model = MyModel("MyModel", 32, 3, 1, 2, 0.5, "relu", 8)
#     model.build(input_shape=(None, 256, 256, 3))
#     model.summary()
