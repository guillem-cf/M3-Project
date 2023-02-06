import numpy as np

import wandb
import matplotlib
from tensorflow.keras.preprocessing.image import ImageDataGenerator

matplotlib.use("Agg")
import matplotlib.pyplot as plt

def preprocess_input(image):
    image = image / 255.0
    image = np.array(image)
    return image

"""
datagen = ImageDataGenerator(
    featurewise_center=True,
    samplewise_center=False,
    featurewise_std_normalization=True,
    samplewise_std_normalization=False,
    preprocessing_function=,
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
"""
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
)


def get_data_train():
    train_generator = datagen.flow_from_directory(
        wandb.config.DATASET_DIR + "/train",
        target_size=(wandb.config.IMG_WIDTH, wandb.config.IMG_HEIGHT),
        batch_size=wandb.config.BATCH_SIZE,
        class_mode="categorical",
    )
    return train_generator


def get_data_validation():
    validation_generator = datagen.flow_from_directory(
        wandb.config.DATASET_DIR + "/train",
        target_size=(wandb.config.IMG_WIDTH, wandb.config.IMG_HEIGHT),
        batch_size=wandb.config.BATCH_SIZE,
        class_mode="categorical",
    )
    return validation_generator


def get_data_test():
    test_generator = datagen.flow_from_directory(
        wandb.config.DATASET_DIR + "/test",
        target_size=(wandb.config.IMG_WIDTH, wandb.config.IMG_HEIGHT),
        batch_size=wandb.config.BATCH_SIZE,
    )
    return test_generator


def sweep(args):
    sweep_config = {
        'method': 'random',
        'name': 'baseline',
        'metric': {'goal': 'maximize', 'name': 'val_accuracy'},
        'parameters':
            {
                'experiment_name': {'value': args.experiment_name},
                'MODEL_FNAME': {'value': args.MODEL_FNAME},
                'DATASET_DIR': {'value': args.DATASET_DIR},
                'LEARNING_RATE': {'value': args.LEARNING_RATE},
                'EPOCHS': {'value': args.EPOCHS},
                'BATCH_SIZE': {'value': args.BATCH_SIZE},
                'OPTIMIZER': {'value': args.OPTIMIZER},
                'LOSS': {'value': args.LOSS},
                'IMG_WIDTH': {'value': args.IMG_WIDTH},
                'IMG_HEIGHT': {'value': args.IMG_HEIGHT},
                'DROPOUT': {'value': args.DROPOUT},
                'WEIGHT_DECAY': {'value': args.WEIGHT_DECAY},
                'VALIDATION_SAMPLES': {'value': args.VALIDATION_SAMPLES}
            }
    }
    return sweep_config


def save_plots(history, args):
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