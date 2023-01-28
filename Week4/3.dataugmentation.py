import matplotlib
import tensorflow as tensorflow
import wandb
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.layers import Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model

import optuna
from optuna.visualization.matplotlib import plot_contour, plot_edf, plot_intermediate_values, plot_optimization_history, plot_parallel_coordinate, plot_param_importances, plot_slice, plot_pareto_front
import os
from optuna.samplers import TPESampler

from tensorflow.python.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from wandb.keras import WandbCallback

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse

print("Num GPUs Available: ", len(tensorflow.config.list_physical_devices("GPU")))
gpus = tensorflow.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tensorflow.config.experimental.set_memory_growth(gpu, True)


parser = argparse.ArgumentParser(
        description="MIT", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
parser.add_argument("--DATASET_DIR", type=str, help="Dataset path", default="./MIT_split")
parser.add_argument(
    "--MODEL_FNAME", type=str, default="./model/full_image/mlp", help="Model path"
)
parser.add_argument(
    "--WEIGHTS_FNAME",
    type=str,
    default="./weights/full_image/mlp",
    help="Weights path",
)
# parser.add_argument("--PATCH_SIZE", type=int, help="Indicate Patch Size", default=64)
parser.add_argument("--BATCH_SIZE", type=int, help="Indicate Batch Size", default=32)
parser.add_argument("--EPOCHS", type=int, help="Indicate Epochs", default=20)
parser.add_argument("--LEARNING_RATE", type=float, help="Indicate Learning Rate", default=0.001)
parser.add_argument("--MOMENTUM", type=float, help="Indicate Momentum", default=0.9)
parser.add_argument("--DROPOUT", type=float, help="Indicate Dropout", default=0)
parser.add_argument("--WEIGHT_DECAY", type=float, help="Indicate Weight Decay", default=0.0001)
parser.add_argument("--OPTIMIZER", type=str, help="Indicate Optimizer", default="sgd")
parser.add_argument(
    "--LOSS", type=str, help="Indicate Loss", default="categorical_crossentropy"
)
parser.add_argument("--IMG_WIDTH", type=int, help="Indicate Image Size", default=224)
parser.add_argument("--IMG_HEIGHT", type=int, help="Indicate Image Size", default=224)
# parser.add_argument("--MODEL", type=int, help="Indicate the model to use", default=1)
parser.add_argument("--experiment_name", type=str, help="Experiment name", default="baseline")
parser.add_argument(
    "--VALIDATION_SAMPLES",
    type=int,
    help="Number of validation samples",
    default=807,
)
parser.add_argument("--horizontal_flip", type=bool, help="Horizontal Flip", default=False)
parser.add_argument("--vertical_flip", type=bool, help="Vertical Flip", default=False)
parser.add_argument("--rotation", type=int, help="Rotation", default=0)
parser.add_argument("--width_shift", type=float, help="Width Shift", default=0.0)
parser.add_argument("--height_shift", type=float, help="Height Shift", default=0.0)
parser.add_argument("--shear_range", type=float, help="Shear Range", default=0.0)
parser.add_argument("--zoom_range", type=float, help="Zoom Range", default=0.0)

args = parser.parse_args()


sweep_config = {
        'method': 'grid',
        'name': 'Task3',
        'metric': {'goal': 'maximize', 'name': 'val_accuracy'},
        'parameters': 
        {
            'experiment_name': {'value': args.experiment_name},
            'MODEL_FNAME':     {'value': args.MODEL_FNAME},
            'DATASET_DIR':     {'value': args.DATASET_DIR},
            'LEARNING_RATE':   {'value': args.LEARNING_RATE},
            'EPOCHS':          {'value': args.EPOCHS},
            'BATCH_SIZE':      {'value': args.BATCH_SIZE},
            'OPTIMIZER':       {'value': args.OPTIMIZER},
            'LOSS':            {'value': args.LOSS},
            'IMG_WIDTH':       {'value': args.IMG_WIDTH},
            'IMG_HEIGHT':      {'value': args.IMG_HEIGHT},
            'DROPOUT':         {'value': args.DROPOUT},
            'WEIGHT_DECAY':    {'value': args.WEIGHT_DECAY},
            'VALIDATION_SAMPLES': {'value': args.VALIDATION_SAMPLES},
            'data_augmentation_HF': {'values': [True, False]},
            'data_augmentation_R': {'values': [0, 20]},#{'max': 20, 'min': 0, 'type': 'int'},
            'data_augmentation_Z': {'values': [0, 0.2]},#{'max': 0.20, 'min': 0.0, 'type': 'double'},
            'data_augmentation_W': {'values': [0, 0.2]},#{'max': 0.20, 'min': 0.0, 'type': 'double'},
            'data_augmentation_H': {'values': [0, 0.2]},#{'max': 0.20, 'min': 0.0, 'type': 'double'},
            'data_augmentation_S': {'values': [0, 0.2]} #{'max': 0.20, 'min': 0.0, 'type': 'double'}
        }   
    }

sweep_id = wandb.sweep(sweep = sweep_config, project="M3_W4")


def train():
    wandb.init(project=args.experiment_name)

    datagen = ImageDataGenerator(
        featurewise_center=True,
        samplewise_center=False,
        featurewise_std_normalization=True,
        samplewise_std_normalization=False,
        preprocessing_function=preprocess_input,
        rotation_range=wandb.config.data_augmentation_R,
        width_shift_range=wandb.config.data_augmentation_W,
        height_shift_range=wandb.config.data_augmentation_H,
        shear_range=wandb.config.data_augmentation_S,
        zoom_range=wandb.config.data_augmentation_Z,
        channel_shift_range=0.0,
        fill_mode="nearest",
        cval=0.0,
        horizontal_flip=wandb.config.data_augmentation_HF,
        vertical_flip=False,
        rescale=None,
    )

    train_generator = datagen.flow_from_directory(
        wandb.config.DATASET_DIR + "/train",
        target_size=(wandb.config.IMG_WIDTH, wandb.config.IMG_HEIGHT),
        batch_size=wandb.config.BATCH_SIZE,
        class_mode="categorical",
    )

    test_generator = datagen.flow_from_directory(
        wandb.config.DATASET_DIR + "/test",
        target_size=(wandb.config.IMG_WIDTH, wandb.config.IMG_HEIGHT),
        batch_size=wandb.config.BATCH_SIZE,
        class_mode="categorical",
    )

    validation_generator = datagen.flow_from_directory(
        wandb.config.DATASET_DIR + "/test",
        target_size=(wandb.config.IMG_WIDTH, wandb.config.IMG_HEIGHT),
        batch_size=wandb.config.BATCH_SIZE,
        class_mode="categorical",
    )

    base_model = DenseNet121(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
    base_model.trainable = False
    # base_model.summary()
    # plot_model(base_model, to_file='modelDenseNet121.png', show_shapes=True, show_layer_names=True)

    # We choose the model 1
    x = base_model.get_layer('pool4_conv').output  # -1 block + -1 transient

    x = GlobalAveragePooling2D()(x)
    # x = Dense(1024, activation='relu')(x)
    x = Dense(8, activation='softmax', name='predictionsProf')(x)

    model = Model(inputs=base_model.input, outputs=x)
    # model.summary()
    # plot_model(model, to_file="modelDenseNet121c.png", show_shapes=True, show_layer_names=True)

    # defining the early stop criteria
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
    # saving the best model based on val_loss
    mc1 = ModelCheckpoint('./checkpoint/best_' + args.experiment_name + '_model_checkpoint' + '.h5',
                          monitor='val_loss', mode='min', save_best_only=True)
    mc2 = ModelCheckpoint('./checkpoint/best_' + args.experiment_name + '_model_checkpoint' + '.h5',
                          monitor='val_accuracy', mode='max', save_best_only=True)
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6)

    model.compile(loss="categorical_crossentropy", optimizer="adadelta", metrics=["accuracy"])

    # preprocessing_function=preprocess_input,

    history = model.fit(
        train_generator,
        steps_per_epoch=(int(400 // wandb.config.BATCH_SIZE) + 1),
        epochs=wandb.config.EPOCHS,
        validation_data=validation_generator,
        validation_steps=(int(wandb.config.VALIDATION_SAMPLES // wandb.config.BATCH_SIZE) + 1),
        callbacks=[WandbCallback(), mc1, mc2],
    )
    # callbacks=[es, mc, mc_2, reduce_lr, WandbCallback()])
    # https://www.tensorflow.org/api_docs/python/tensorflow/keras/callbacks/ReduceLROnPlateau
    # https://keras.io/api/callbacks/model_checkpoint/

    result = model.evaluate(test_generator)
    print(result)
    print(history.history.keys())

    # list all data in history

    if True:
        # summarize history for accuracy
        plt.plot(history.history["accuracy"])
        plt.plot(history.history["val_accuracy"])
        plt.title("model accuracy")
        plt.ylabel("accuracy")
        plt.xlabel("epoch")
        plt.legend(["train", "validation"], loc="upper left")
        plt.savefig("images/accuracy_" + args.experiment_name + ".jpg")
        plt.close()
        # summarize history for loss
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.title("model loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(["train", "validation"], loc="upper left")
        plt.savefig("images/loss_" + args.experiment_name + ".jpg")

    
    wandb.log({
        'train_acc': history.history["accuracy"],
        'train_loss': history.history["loss"], 
        'val_acc': history.history["val_accuracy"], 
        'val_loss': history.history["val_loss"]
    })



# def main():


    # """
    # train_data_dir='/ghome/mcv/datasets/MIT_split/train'
    # val_data_dir='/ghome/mcv/datasets/MIT_split/test'
    # test_data_dir='/ghome/mcv/datasets/MIT_split/test'
    # img_width = 224
    # img_height=224
    # batch_size=32
    # number_of_epoch=2
    # validation_samples=807
    # """
    """
    parser = argparse.ArgumentParser(
        description="MIT", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--DATASET_DIR", type=str, help="Dataset path", default="./MIT_split")
    parser.add_argument(
        "--MODEL_FNAME", type=str, default="./model/full_image/mlp", help="Model path"
    )
    parser.add_argument(
        "--WEIGHTS_FNAME",
        type=str,
        default="./weights/full_image/mlp",
        help="Weights path",
    )
    # parser.add_argument("--PATCH_SIZE", type=int, help="Indicate Patch Size", default=64)
    parser.add_argument("--BATCH_SIZE", type=int, help="Indicate Batch Size", default=32)
    parser.add_argument("--EPOCHS", type=int, help="Indicate Epochs", default=20)
    parser.add_argument("--LEARNING_RATE", type=float, help="Indicate Learning Rate", default=0.001)
    parser.add_argument("--MOMENTUM", type=float, help="Indicate Momentum", default=0.9)
    parser.add_argument("--DROPOUT", type=float, help="Indicate Dropout", default=0)
    parser.add_argument("--WEIGHT_DECAY", type=float, help="Indicate Weight Decay", default=0.0001)
    parser.add_argument("--OPTIMIZER", type=str, help="Indicate Optimizer", default="sgd")
    parser.add_argument(
        "--LOSS", type=str, help="Indicate Loss", default="categorical_crossentropy"
    )
    parser.add_argument("--IMG_WIDTH", type=int, help="Indicate Image Size", default=224)
    parser.add_argument("--IMG_HEIGHT", type=int, help="Indicate Image Size", default=224)
    # parser.add_argument("--MODEL", type=int, help="Indicate the model to use", default=1)
    parser.add_argument("--experiment_name", type=str, help="Experiment name", default="baseline")
    parser.add_argument(
        "--VALIDATION_SAMPLES",
        type=int,
        help="Number of validation samples",
        default=807,
    )
    parser.add_argument("--horizontal_flip", type=bool, help="Horizontal Flip", default=False)
    parser.add_argument("--vertical_flip", type=bool, help="Vertical Flip", default=False)
    parser.add_argument("--rotation", type=int, help="Rotation", default=0)
    parser.add_argument("--width_shift", type=float, help="Width Shift", default=0.0)
    parser.add_argument("--height_shift", type=float, help="Height Shift", default=0.0)
    parser.add_argument("--shear_range", type=float, help="Shear Range", default=0.0)
    parser.add_argument("--zoom_range", type=float, help="Zoom Range", default=0.0)

    args = parser.parse_args()


    sweep_config = {
            'method': 'grid',
            'name': 'Task3',
            'metric': {'goal': 'maximize', 'name': 'val_accuracy'},
            'parameters': 
            {
                'experiment_name': {'value': args.experiment_name},
                'MODEL_FNAME':     {'value': args.MODEL_FNAME},
                'DATASET_DIR':     {'value': args.DATASET_DIR},
                'LEARNING_RATE':   {'value': args.LEARNING_RATE},
                'EPOCHS':          {'value': args.EPOCHS},
                'BATCH_SIZE':      {'value': args.BATCH_SIZE},
                'OPTIMIZER':       {'value': args.OPTIMIZER},
                'LOSS':            {'value': args.LOSS},
                'IMG_WIDTH':       {'value': args.IMG_WIDTH},
                'IMG_HEIGHT':      {'value': args.IMG_HEIGHT},
                'DROPOUT':         {'value': args.DROPOUT},
                'WEIGHT_DECAY':    {'value': args.WEIGHT_DECAY},
                'VALIDATION_SAMPLES': {'value': args.VALIDATION_SAMPLES},
                'data_augmentation_HF': {'values': [True, False]},
                'data_augmentation_R': {'values': [0, 20]},#{'max': 20, 'min': 0, 'type': 'int'},
                'data_augmentation_Z': {'values': [0, 0.2]},#{'max': 0.20, 'min': 0.0, 'type': 'double'},
                'data_augmentation_W': {'values': [0, 0.2]},#{'max': 0.20, 'min': 0.0, 'type': 'double'},
                'data_augmentation_H': {'values': [0, 0.2]},#{'max': 0.20, 'min': 0.0, 'type': 'double'},
                'data_augmentation_S': {'values': [0, 0.2]} #{'max': 0.20, 'min': 0.0, 'type': 'double'}
            }   
        }

    sweep_id = wandb.sweep(sweep = sweep_config, project="M3_W4")
    """

wandb.agent(sweep_id, function=train, count=37)
