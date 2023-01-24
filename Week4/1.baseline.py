import tensorflow as tf
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard
import matplotlib
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import wandb
from wandb.keras import WandbCallback

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def preprocess_input(x, dim_ordering='default'):
    if dim_ordering == 'default':
        dim_ordering = K.image_data_format()
    assert dim_ordering in {'channels_first', 'channels_last'}

    if dim_ordering == 'channels_first':
        # 'RGB'->'BGR'
        x = x[::-1, :, :]
        # Zero-center by mean pixel
        x[0, :, :] -= 103.939
        x[1, :, :] -= 116.779
        x[2, :, :] -= 123.68
    else:
        # 'RGB'->'BGR'
        x = x[:, :, ::-1]
        # Zero-center by mean pixel
        x[:, :, 0] -= 103.939
        x[:, :, 1] -= 116.779
        x[:, :, 2] -= 123.68
    return x


def train(args):
    datagen = ImageDataGenerator(featurewise_center=False,
                                 samplewise_center=False,
                                 featurewise_std_normalization=False,
                                 samplewise_std_normalization=False,
                                 preprocessing_function=preprocess_input,
                                 rotation_range=0.,
                                 width_shift_range=0.,
                                 height_shift_range=0.,
                                 shear_range=0.,
                                 zoom_range=0.,
                                 channel_shift_range=0.,
                                 fill_mode='nearest',
                                 cval=0.,
                                 horizontal_flip=False,
                                 vertical_flip=False,
                                 rescale=None)

    train_generator = datagen.flow_from_directory(args.DATASET_DIR + '/train',
                                                  target_size=(args.IMG_WIDTH, args.IMG_HEIGHT),
                                                  batch_size=args.BATCH_SIZE,
                                                  class_mode='categorical')

    test_generator = datagen.flow_from_directory(args.DATASET_DIR + '/test',
                                                 target_size=(args.IMG_WIDTH, args.IMG_HEIGHT),
                                                 batch_size=args.BATCH_SIZE,
                                                 class_mode='categorical')

    validation_generator = datagen.flow_from_directory(args.DATASET_DIR + '/test',
                                                       target_size=(args.IMG_WIDTH, args.IMG_HEIGHT),
                                                       batch_size=args.BATCH_SIZE,
                                                       class_mode='categorical')

    base_model = DenseNet121(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    base_model.trainable = False
    # base_model.summary()
    # plot_model(base_model, to_file='modelDenseNet121.png', show_shapes=True, show_layer_names=True)

    x = Flatten()(base_model.output)
    # x = Dense(512, activation='relu')(x)
    # x = Dense(256, activation='relu')(x)
    output = Dense(8, activation='softmax', name='predictions')(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.summary()
    plot_model(model, to_file='modelDenseNet121c.png', show_shapes=True, show_layer_names=True)

    # defining the early stop criteria
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
    # saving the best model based on val_loss
    mc = ModelCheckpoint('./checkpoint/best_' + args.experiment_name + '_model_checkpoint' + '.h5',
                         monitor='val_loss', mode='min', save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6)

    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    # preprocessing_function=preprocess_input,

    history = model.fit(train_generator,
                        steps_per_epoch=(int(400 // args.BATCH_SIZE) + 1),
                        epochs=args.EPOCHS,
                        validation_data=validation_generator,
                        validation_steps=(int(args.VALIDATION_SAMPLES // args.BATCH_SIZE) + 1),
                        callbacks=[WandbCallback()])
    # callbacks=[es, mc, mc_2, reduce_lr, WandbCallback()])
    # https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ReduceLROnPlateau
    # https://keras.io/api/callbacks/model_checkpoint/

    result = model.evaluate(test_generator)
    print(result)
    print(history.history.keys())

    # list all data in history

    if True:
        # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig("images/accuracy_" + args.experiment_name + ".jpg")
        plt.close()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig("images/loss_" + args.experiment_name + ".jpg")
    wandb.finish()


if __name__ == "__main__":
    """
    train_data_dir='/ghome/mcv/datasets/MIT_split/train'
    val_data_dir='/ghome/mcv/datasets/MIT_split/test'
    test_data_dir='/ghome/mcv/datasets/MIT_split/test'
    img_width = 224
    img_height=224
    batch_size=32
    number_of_epoch=2
    validation_samples=807
    """

    parser = argparse.ArgumentParser(description="MIT", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--DATASET_DIR", type=str, help="Dataset path", default="/ghome/group07/mcv/datasets/MIT_split")
    parser.add_argument("--MODEL_FNAME", type=str, default="./model/full_image/mlp", help="Model path")
    parser.add_argument("--WEIGHTS_FNAME", type=str, default="./weights/full_image/mlp", help="Weights path")
    # parser.add_argument("--PATCH_SIZE", type=int, help="Indicate Patch Size", default=64)
    parser.add_argument("--BATCH_SIZE", type=int, help="Indicate Batch Size", default=32)
    parser.add_argument("--EPOCHS", type=int, help="Indicate Epochs", default=20)
    parser.add_argument("--LEARNING_RATE", type=float, help="Indicate Learning Rate", default=0.001)
    parser.add_argument("--MOMENTUM", type=float, help="Indicate Momentum", default=0.9)
    parser.add_argument("--DROPOUT", type=float, help="Indicate Dropout", default=0)
    parser.add_argument("--WEIGHT_DECAY", type=float, help="Indicate Weight Decay", default=0.0001)
    parser.add_argument("--OPTIMIZER", type=str, help="Indicate Optimizer", default="sgd")
    parser.add_argument("--LOSS", type=str, help="Indicate Loss", default="categorical_crossentropy")
    parser.add_argument("--IMG_WIDTH", type=int, help="Indicate Image Size", default=224)
    parser.add_argument("--IMG_HEIGHT", type=int, help="Indicate Image Size", default=224)
    # parser.add_argument("--MODEL", type=int, help="Indicate the model to use", default=1)
    parser.add_argument("--experiment_name", type=str, help="Experiment name", default="baseline")
    parser.add_argument("--VALIDATION_SAMPLES", type=int, help="Number of validation samples", default=807)
    args = parser.parse_args()

    config = dict(
        model_name=args.experiment_name,
        learning_rate=args.LEARNING_RATE,
        momentum=args.MOMENTUM,
        architecture="MLP",
        dataset="MIT",
        optimizer=args.OPTIMIZER,  # sgd, adam, rmsprop
        loss=args.LOSS,
        image_width=args.IMG_WIDTH,
        image_height=args.IMG_HEIGHT,
        batch_size=args.BATCH_SIZE,
        epochs=args.EPOCHS,
        validation_samples=args.VALIDATION_SAMPLES,
        weight_decay=args.WEIGHT_DECAY,
        dropout=args.DROPOUT,
        # model = args.MODEL,
    )

    wandb.init(
        project="M3_W4",
        config=config,
        name=args.experiment_name,
    )

    train(args)
