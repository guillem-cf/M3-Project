import matplotlib
import numpy as np
import tensorflow as tf
import wandb
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2


matplotlib.use("Agg")
import matplotlib.pyplot as plt

def blur(img):
    return np.array(cv2.GaussianBlur(img, (5, 5), 1.0))


def get_data_train():
    datagen = ImageDataGenerator(
        preprocessing_function=blur,
        rotation_range=wandb.config.rotation,
        width_shift_range=wandb.config.width_shift,
        height_shift_range=wandb.config.height_shift,
        shear_range=wandb.config.shear_range,
        zoom_range=wandb.config.zoom_range,
        horizontal_flip=wandb.config.horizontal_flip,
        vertical_flip=wandb.config.vertical_flip,
        brightness_range=wandb.config.brightness_range,
        rescale=1. / 255
    )
    train_generator = datagen.flow_from_directory(
        wandb.config.DATASET_DIR + "/train",
        target_size=(wandb.config.IMG_WIDTH, wandb.config.IMG_HEIGHT),
        batch_size=wandb.config.BATCH_SIZE,
        class_mode="categorical",
    )
    return train_generator


def get_data_validation():
    # datagen = ImageDataGenerator(
    #     preprocessing_function=blur,
    #     rotation_range=wandb.config.rotation,
    #     width_shift_range=wandb.config.width_shift,
    #     height_shift_range=wandb.config.height_shift,
    #     shear_range=wandb.config.shear_range,
    #     zoom_range=wandb.config.zoom_range,
    #     horizontal_flip=wandb.config.horizontal_flip,
    #     vertical_flip=wandb.config.vertical_flip,
    #     brightness_range=wandb.config.brightness_range,
    #     rescale=1. / 255
    # )
    datagen = ImageDataGenerator(rescale=1. / 255)
    validation_generator = datagen.flow_from_directory(
        wandb.config.DATASET_DIR + "/test",
        target_size=(wandb.config.IMG_WIDTH, wandb.config.IMG_HEIGHT),
        batch_size=wandb.config.BATCH_SIZE,
        class_mode="categorical",
    )
    return validation_generator


def get_data_test():
    datagen_test = ImageDataGenerator(rescale=1. / 255)
    test_generator = datagen_test.flow_from_directory(
        wandb.config.DATASET_DIR + "/test",
        target_size=(wandb.config.IMG_WIDTH, wandb.config.IMG_HEIGHT),
        batch_size=wandb.config.BATCH_SIZE,
    )
    return test_generator



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