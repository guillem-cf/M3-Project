import matplotlib
import numpy as np
import tensorflow as tf
import wandb
from tensorflow.keras.preprocessing.image import ImageDataGenerator

matplotlib.use("Agg")
import matplotlib.pyplot as plt


datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    channel_shift_range=0.0,
    fill_mode="nearest",
    cval=0.0,
    horizontal_flip=True,
    vertical_flip=False,
    rescale=1. / 255,
)
datagen_test = ImageDataGenerator(rescale=1. / 255)
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
        wandb.config.DATASET_DIR + "/test",
        target_size=(wandb.config.IMG_WIDTH, wandb.config.IMG_HEIGHT),
        batch_size=wandb.config.BATCH_SIZE,
        class_mode="categorical",
    )
    return validation_generator


def get_data_test():
    test_generator = datagen_test.flow_from_directory(
        wandb.config.DATASET_DIR + "/test",
        target_size=(wandb.config.IMG_WIDTH, wandb.config.IMG_HEIGHT),
        batch_size=wandb.config.BATCH_SIZE,
    )
    return test_generator


def sweep(args):
    sweep_config = {
        'method': 'random',
        'name': 'Sweep_' + args.experiment_name,
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
                'VALIDATION_SAMPLES': {'value': args.VALIDATION_SAMPLES},
                'data_augmentation_HF': {'value': True},
                'data_augmentation_Z': {'value': 0.2},
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

def get_optimizer(optimizer="adam"):
    "Select optmizer between adam and sgd with momentum"
    if optimizer.lower() == "adam":
        return tf.keras.optimizers.Adam(learning_rate=wandb.config.LEARNING_RATE,
                                        weight_decay=wandb.config.WEIGHT_DECAY)
    if optimizer.lower() == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=wandb.config.LEARNING_RATE, momentum=wandb.config.MOMENTUM,
                                       weight_decay=wandb.config.WEIGHT_DECAY)
    if optimizer.lower() == "rmsprop":
        return tf.keras.optimizers.RMSprop(learning_rate=wandb.config.LEARNING_RATE, momentum=wandb.config.MOMENTUM,
                                           weight_decay=wandb.config.WEIGHT_DECAY)
    if optimizer.lower() == "adagrad":
        return tf.keras.optimizers.Adagrad(learning_rate=wandb.config.LEARNING_RATE,
                                           weight_decay=wandb.config.WEIGHT_DECAY)
    if optimizer.lower() == "adadelta":
        return tf.keras.optimizers.Adadelta(learning_rate=wandb.config.LEARNING_RATE,
                                            weight_decay=wandb.config.WEIGHT_DECAY)
    if optimizer.lower() == "adamax":
        return tf.keras.optimizers.Adamax(learning_rate=wandb.config.LEARNING_RATE,
                                          weight_decay=wandb.config.WEIGHT_DECAY)
    if optimizer.lower() == "nadam":
        return tf.keras.optimizers.Nadam(learning_rate=wandb.config.LEARNING_RATE,
                                         weight_decay=wandb.config.WEIGHT_DECAY)