import argparse

import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import wandb
from tensorflow.keras import layers
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from wandb.keras import WandbCallback

matplotlib.use('Agg')
"""
print("Num GPUs Available: ", len(tensorflow.config.list_physical_devices('GPU')))
gpus = tensorflow.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tensorflow.config.experimental.set_memory_growth(gpu, True)
"""


def train(args):
    datagen = ImageDataGenerator(rescale=None, horizontal_flip=True, preprocessing_function=preprocess_input)

    train_generator = datagen.flow_from_directory(args.DATASET_DIR + '/train',
                                                  target_size=(
                                                      args.IMG_WIDTH, args.IMG_HEIGHT),
                                                  batch_size=args.BATCH_SIZE,
                                                  class_mode='categorical')

    test_generator = datagen.flow_from_directory(args.DATASET_DIR + '/test',
                                                 target_size=(
                                                     args.IMG_WIDTH, args.IMG_HEIGHT),
                                                 batch_size=args.BATCH_SIZE,
                                                 class_mode='categorical')

    validation_generator = datagen.flow_from_directory(args.DATASET_DIR + '/test',
                                                       target_size=(
                                                           args.IMG_WIDTH, args.IMG_HEIGHT),
                                                       batch_size=args.BATCH_SIZE,
                                                       class_mode='categorical')
    model = tf.keras.models.load_model('./checkpoint/' + args.MODEL_NAME)
    plot_model(model, to_file='./images/all_model.png', show_shapes=True, show_layer_names=True)
    model.trainable = True
    for layer in model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        print(layer.name, layer.trainable)
    model.summary()

    # defining the early stop criteria
    EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
    # saving the best model based on val_loss
    mc = ModelCheckpoint('./checkpoint/bestloss_' + args.experiment_name + '_model_checkpoint' + '.h5',
                         monitor='val_loss', mode='min', save_best_only=True)
    mc2 = ModelCheckpoint('./checkpoint/bestaccuracy_' + args.experiment_name + '_model_checkpoint' + '.h5',
                          monitor='val_accuracy', mode='max', save_best_only=True)
    ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta', metrics=['accuracy'])

    history = model.fit(train_generator,
                        steps_per_epoch=(int(400 // args.BATCH_SIZE) + 1),
                        epochs=args.EPOCHS,
                        validation_data=validation_generator,
                        validation_steps=(
                                int(args.VALIDATION_SAMPLES // args.BATCH_SIZE) + 1),
                        callbacks=[WandbCallback(), mc, mc2])

    result = model.evaluate(test_generator)
    print(result)
    print(history.history.keys())

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

    parser = argparse.ArgumentParser(
        description="MIT", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--DATASET_DIR", type=str, help="Dataset path",
                        default="/ghome/mcv/datasets/MIT_split")
    parser.add_argument("--MODEL_FNAME", type=str,
                        default="./model/full_image/mlp", help="Model path")
    parser.add_argument("--WEIGHTS_FNAME", type=str,
                        default="./weights/full_image/mlp", help="Weights path")
    # parser.add_argument("--PATCH_SIZE", type=int, help="Indicate Patch Size", default=64)
    parser.add_argument("--BATCH_SIZE", type=int,
                        help="Indicate Batch Size", default=32)
    parser.add_argument("--EPOCHS", type=int,
                        help="Indicate Epochs", default=50)
    parser.add_argument("--LEARNING_RATE", type=float,
                        help="Indicate Learning Rate", default=0.001)
    parser.add_argument("--MOMENTUM", type=float,
                        help="Indicate Momentum", default=0.9)
    parser.add_argument("--DROPOUT", type=float,
                        help="Indicate Dropout", default=0)
    parser.add_argument("--WEIGHT_DECAY", type=float,
                        help="Indicate Weight Decay", default=0.0001)
    parser.add_argument("--OPTIMIZER", type=str,
                        help="Indicate Optimizer", default="sgd")
    parser.add_argument("--LOSS", type=str, help="Indicate Loss",
                        default="categorical_crossentropy")
    parser.add_argument("--IMG_WIDTH", type=int,
                        help="Indicate Image Size", default=224)
    parser.add_argument("--IMG_HEIGHT", type=int,
                        help="Indicate Image Size", default=224)
    # parser.add_argument("--MODEL", type=int, help="Indicate the model to use", default=1)
    parser.add_argument("--experiment_name", type=str,
                        help="Experiment name", default="baseline")
    parser.add_argument("--VALIDATION_SAMPLES", type=int,
                        help="Number of validation samples", default=807)
    parser.add_argument("--MODEL_START", type=int,
                        help="1: flatten, 2:GAP", default=1)
    parser.add_argument("--MODEL_HID", nargs="+", type=int, help="Indicate the model to use", default=None)
    parser.add_argument("--MODEL_NAME", type=str, help="Indicate name model to use", default=None)

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
