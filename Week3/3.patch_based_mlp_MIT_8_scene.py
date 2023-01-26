from __future__ import print_function

import argparse

import matplotlib.pyplot as plt
import tensorflow as tf
import wandb
from tensorflow.keras.layers import Dense, Reshape
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from wandb.keras import WandbCallback

from utils import *

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

parser = argparse.ArgumentParser(description="MIT", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--DATASET_DIR", type=str, help="Dataset path", default="./MIT_split")
parser.add_argument("--PATCHES_DIR", type=str, help="Patches path", default="./MIT_split_patches")
parser.add_argument("--MODEL_FNAME", type=str, default="./model/patch_based/patch_based_mlp_", help="Model path")
parser.add_argument("--WEIGHTS_FNAME", type=str, default="./weights/patch_based/patch_based_mlp_", help="Weights path")
parser.add_argument("--PATCH_SIZE", type=int, help="Indicate Patch Size", default=64)
parser.add_argument("--BATCH_SIZE", type=int, help="Indicate Batch Size", default=16)
parser.add_argument("--EPOCHS", type=int, help="Indicate Epochs", default=100)
parser.add_argument("--LEARNING_RATE", type=float, help="Indicate Learning Rate", default=0.001)
parser.add_argument("--MOMENTUM", type=float, help="Indicate Momentum", default=0.9)
parser.add_argument("--DROPOUT", type=float, help="Indicate Dropout", default=0)
parser.add_argument("--WEIGHT_DECAY", type=float, help="Indicate Weight Decay", default=0.0001)
parser.add_argument("--OPTIMIZER", type=str, help="Indicate Optimizer", default="sgd")
parser.add_argument("--LOSS", type=str, help="Indicate Loss", default="categorical_crossentropy")
parser.add_argument("--IMG_SIZE", type=int, help="Indicate Image Size", default=32)
parser.add_argument("--MODEL", type=int, help="Indicate the model to use", default=1)
parser.add_argument("--experiment_name", type=str, help="Experiment name", default="baseline")
parser.add_argument("--AGGREGATION", type=str, help="Aggregation block", default="mean")
args = parser.parse_args()

# user defined variables
"""
PATCH_SIZE = 64
BATCH_SIZE = 16
DATASET_DIR = '/home/mcv/datasets/MIT_split'
PATCHES_DIR = '/home/group10/m3/data/MIT_split_patches' + str(PATCH_SIZE)
MODEL_FNAME = '/home/group10/m3/patch_based_mlp.h5'
"""

config = dict(
    model_name=args.experiment_name,
    learning_rate=args.LEARNING_RATE,
    momentum=args.MOMENTUM,
    architecture="MLP",
    dataset="MIT",
    optimizer=args.OPTIMIZER,  # sgd, adam, rmsprop
    loss=args.LOSS,
    image_size=args.IMG_SIZE,
    batch_size=args.BATCH_SIZE,
    epochs=args.EPOCHS,
    weight_decay=args.WEIGHT_DECAY,
    dropout=args.DROPOUT,
    model=args.MODEL,
)

wandb.init(
    project="M3",
    tags=[args.experiment_name],
    config=config,
)

PATCHES_DIR = args.PATCHES_DIR + str(args.PATCH_SIZE)


def build_mlp(input_size=args.PATCH_SIZE, phase="TRAIN"):
    model = Sequential()
    model.add(
        Reshape((input_size * input_size * 3,), input_shape=(input_size, input_size, 3), name='first',
                dtype='float32'))
    initializer = tf.keras.initializers.RandomNormal()
    model.add(Dense(units=3072, activation="relu", kernel_initializer=initializer, name="second"))
    model.add(Dense(units=2048, activation="relu", kernel_initializer=initializer, name="third"))
    model.add(Dense(units=1024, activation="relu", kernel_initializer=initializer, name="fourth"))
    model.add(Dense(units=256, activation="relu", kernel_initializer=initializer, name="fifth"))

    if phase == "TEST":
        model.add(
            Dense(units=8, activation="linear")
        )  # In test phase we softmax the average output over the image patches
    else:
        model.add(Dense(units=8, activation="softmax"))

    return model


if not os.path.exists(args.DATASET_DIR):
    print("ERROR: dataset directory " + args.DATASET_DIR + " do not exists!\n")
    quit()
if not os.path.exists(PATCHES_DIR):
    print("WARNING: patches dataset directory " + PATCHES_DIR + " does not exist!\n")
    print("Creating image patches dataset into " + PATCHES_DIR + "\n")
    generate_image_patches_db(args.DATASET_DIR, PATCHES_DIR, patch_size=args.PATCH_SIZE)
    print("Done!\n")

print("Building MLP model...\n")

model = build_mlp(input_size=args.PATCH_SIZE)

model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

print(model.summary())

print("Done!\n")

train = True
if not os.path.exists(args.WEIGHTS_FNAME + args.experiment_name + ".h5") or train:
    print("WARNING: model file " + args.WEIGHTS_FNAME + args.experiment_name + ".h5 do not exists!\n")
    print("Start training...\n")
    # this is the dataset configuration we will use for training
    # only rescaling
    train_datagen = ImageDataGenerator(rescale=1.0 / 255, horizontal_flip=True)

    # this is the dataset configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    # this is a generator that will read pictures found in
    # subfolers of 'data/train', and indefinitely generate
    # batches of augmented image data
    train_generator = train_datagen.flow_from_directory(
        PATCHES_DIR + "/train",  # this is the target directory
        target_size=(
            args.PATCH_SIZE,
            args.PATCH_SIZE,
        ),  # all images will be resized to PATCH_SIZExPATCH_SIZE
        batch_size=args.BATCH_SIZE,
        classes=[
            "coast",
            "forest",
            "highway",
            "inside_city",
            "mountain",
            "Opencountry",
            "street",
            "tallbuilding",
        ],
        class_mode="categorical",
    )  # since we use binary_crossentropy loss, we need categorical labels

    # this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
        PATCHES_DIR + "/test",
        target_size=(args.PATCH_SIZE, args.PATCH_SIZE),
        batch_size=args.BATCH_SIZE,
        classes=[
            "coast",
            "forest",
            "highway",
            "inside_city",
            "mountain",
            "Opencountry",
            "street",
            "tallbuilding",
        ],
        class_mode="categorical",
    )

    history = model.fit(
        train_generator,
        steps_per_epoch=18810 // args.BATCH_SIZE,
        epochs=150,
        validation_data=validation_generator,
        validation_steps=8070 // args.BATCH_SIZE,
        callbacks=[WandbCallback()]
    )

    print("Done!\n")
    print("Saving the model into " + args.WEIGHTS_FNAME + " \n")
    model.save_weights(
        args.WEIGHTS_FNAME + args.experiment_name + ".h5")  # always save your weights after training or during training
    model.save(
        args.MODEL_FNAME + args.experiment_name + ".h5")
    print("Done!\n")

    # summarize history for accuracy
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.savefig("images/patch_based/accuracy_" + args.experiment_name + ".jpg")
    plt.close()
    # summarize history for loss
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.savefig("images/patch_based/loss_" + args.experiment_name + ".jpg")

print("Building MLP model for testing...\n")

model = build_mlp(input_size=args.PATCH_SIZE, phase="TEST")
print(model.summary())

print("Done!\n")

print("Loading weights from " + args.WEIGHTS_FNAME + " ...\n")
print("\n")

model.load_weights(args.WEIGHTS_FNAME + args.experiment_name + ".h5")

print("Done!\n")

print("Start evaluation ...\n")

directory = args.DATASET_DIR + "/test"
classes = {
    "coast": 0,
    "forest": 1,
    "highway": 2,
    "inside_city": 3,
    "mountain": 4,
    "Opencountry": 5,
    "street": 6,
    "tallbuilding": 7,
}
correct = 0.0
total = 807
count = 0

for class_dir in os.listdir(directory):
    cls = classes[class_dir]
    for imname in os.listdir(os.path.join(directory, class_dir)):
        im = Image.open(os.path.join(directory, class_dir, imname))
        patches = image.extract_patches_2d(
            np.array(im), (args.PATCH_SIZE, args.PATCH_SIZE),
            max_patches=(int(np.asarray(im).shape[0] / args.PATCH_SIZE) ** 2)
        )
        out = model.predict(patches / 255.0)
        if args.AGGREGATION == "mean":
            predicted_cls = np.argmax(softmax(np.mean(out, axis=0)))
        elif args.AGGREGATION == "max":
            predicted_cls = np.argmax(softmax(np.max(out, axis=0)))
        elif args.AGGREGATION == "min":
            predicted_cls = np.argmax(softmax(np.min(out, axis=0)))
        else:
            raise Exception("Unknown aggregation method!")
        if predicted_cls == cls:
            correct += 1
        count += 1
        print("Evaluated images: " + str(count) + " / " + str(total), end="\r")

print("Done!\n")
print("Test Acc. = " + str(correct / total) + "\n")
wandb.finish()
