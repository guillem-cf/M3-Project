import argparse

import matplotlib
import tensorflow as tf
import wandb
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten, Dense, Reshape, BatchNormalization, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
from wandb.keras import WandbCallback

from utils import *

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

parser = argparse.ArgumentParser(description="MIT", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--DATASET_DIR", type=str, help="Dataset path", default="./MIT_split")
parser.add_argument("--PATCHES_DIR", type=str, help="Patches path", default="./MIT_split_patches")
parser.add_argument("--MODEL_FNAME", type=str, default="./model/full_image/mlp", help="Model path")
parser.add_argument("--WEIGHTS_FNAME", type=str, default="./weights/full_image/mlp", help="Weights path")
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
args = parser.parse_args()

# user defined variables
"""
IMG_SIZE = 32
BATCH_SIZE = 16
DATASET_DIR = '/home/mcv/datasets/MIT_split'
MODEL_FNAME = '/home/group10/m3/my_first_mlp.h5'
"""

config = dict(
    model_name = args.experiment_name,
    learning_rate=args.LEARNING_RATE,
    momentum=args.MOMENTUM,
    architecture="MLP",
    dataset="MIT",
    optimizer=args.OPTIMIZER,  # sgd, adam, rmsprop
    loss=args.LOSS,
    image_size = args.IMG_SIZE,
    batch_size = args.BATCH_SIZE,
    epochs = args.EPOCHS,
    weight_decay = args.WEIGHT_DECAY,
    dropout = args.DROPOUT,
    model = args.MODEL,
)

wandb.init(
    project="M3",
    config=config,
    name=args.experiment_name,
)

PATCHES_DIR = args.PATCHES_DIR + str(args.PATCH_SIZE)

if not os.path.exists(args.DATASET_DIR):
    print("ERROR: dataset directory " + args.DATASET_DIR + " does not exist!\n")
    quit()

print("Building MLP model...\n")

# Build the Multi Layer Perceptron model

if(args.MODEL == 0): # initial model

    model = Sequential()
    model.add(
        Reshape((args.IMG_SIZE * args.IMG_SIZE * 3,), input_shape=(args.IMG_SIZE, args.IMG_SIZE, 3), name='first',
                dtype='float32'))
    model.add(Dense(units=2048, activation="relu", name="second"))
    model.add(Dense(units=8, activation="softmax"))

elif(args.MODEL == 1): # adding batch normalization
    
            model = Sequential()
            model.add(
                Reshape((args.IMG_SIZE * args.IMG_SIZE * 3,), input_shape=(args.IMG_SIZE, args.IMG_SIZE, 3), name='first',
                        dtype='float32'))
            model.add(Dense(units=2048, activation="relu", name="second"))
            model.add(BatchNormalization())
            model.add(Dense(units=1024, activation="relu", name="third"))
            model.add(BatchNormalization())
            model.add(Dense(units=512, activation="relu", name="fourth"))
            model.add(BatchNormalization())
            model.add(Dense(units=256, activation="relu", name="fifth"))
            model.add(BatchNormalization())
            model.add(Dense(units=8, activation="softmax"))

elif(args.MODEL == 2): # adding Dropuout
    
            model = Sequential()
            model.add(
                Reshape((args.IMG_SIZE * args.IMG_SIZE * 3,), input_shape=(args.IMG_SIZE, args.IMG_SIZE, 3), name='first',
                        dtype='float32'))
            model.add(Dense(units=2048, activation="relu", name="second"))
            model.add(Dropout(0.5))
            model.add(Dense(units=1024, activation="relu", name="third"))
            model.add(Dropout(0.5))
            model.add(Dense(units=512, activation="relu", name="fourth"))
            model.add(Dropout(0.5))
            model.add(Dense(units=256, activation="relu", name="fifth"))
            model.add(Dropout(0.5))
            model.add(Dense(units=8, activation="softmax"))

elif(args.MODEL == 3): # adding batch normalization and dropout
    
            model = Sequential()
            model.add(
                Reshape((args.IMG_SIZE * args.IMG_SIZE * 3,), input_shape=(args.IMG_SIZE, args.IMG_SIZE, 3), name='first',
                        dtype='float32'))
            model.add(Dense(units=2048, activation="relu", name="second"))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))
            model.add(Dense(units=1024, activation="relu", name="third"))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))
            model.add(Dense(units=512, activation="relu", name="fourth"))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))
            model.add(Dense(units=256, activation="relu", name="fifth"))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))
            model.add(Dense(units=8, activation="softmax"))

elif(args.MODEL == 4): # starting with a bigger layer and 4 hidden layers 
            model = Sequential()
            model.add(
                Reshape((args.IMG_SIZE * args.IMG_SIZE * 3,), input_shape=(args.IMG_SIZE, args.IMG_SIZE, 3), name='first',
                        dtype='float32'))
            initializer = tf.keras.initializers.GlorotNormal() #default
                    
            model.add(Dense(units=4096, activation="relu",kernel_initializer=initializer, name="second"))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))
            
            model.add(Dense(units=2048, activation="relu",kernel_initializer=initializer, name="second"))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))

            model.add(Dense(units=1024, activation="relu",kernel_initializer=initializer, name="second"))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))

            model.add(Dense(units=512, activation="relu",kernel_initializer=initializer,name="third"))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))

               
            model.add(Dense(units=8, activation="softmax"))


elif(args.MODEL == 5): # starting with a bigger layer and 4 hidden layers
            model = Sequential()
            model.add(
                Reshape((args.IMG_SIZE * args.IMG_SIZE * 3,), input_shape=(args.IMG_SIZE, args.IMG_SIZE, 3), name='first',
                        dtype='float32'))
            #initializer = tf.keras.initializers.GlorotNormal() #default
            initializer = tf.keras.initializers.RandomNormal()
            #initializer = tf.keras.initializers.RandomUniform()
            #initializer = tf.keras.initializers.TruncatedNormal()
                    
            model.add(Dense(units=4096, activation="relu",kernel_initializer=initializer, name="second"))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))
            
            model.add(Dense(units=2048, activation="relu",kernel_initializer=initializer, name="third"))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))

            model.add(Dense(units=1024, activation="relu",kernel_initializer=initializer, name="fourth"))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))

            model.add(Dense(units=256, activation="relu",kernel_initializer=initializer,name="fifht"))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))

               
            model.add(Dense(units=8, activation="softmax"))

elif(args.MODEL == 6): # 4 hidden layers with two of the same size
            model = Sequential()
            model.add(
                Reshape((args.IMG_SIZE * args.IMG_SIZE * 3,), input_shape=(args.IMG_SIZE, args.IMG_SIZE, 3), name='first',
                        dtype='float32'))
            initializer = tf.keras.initializers.GlorotNormal() #default
                    
            model.add(Dense(units=4096, activation="relu",kernel_initializer=initializer, name="second"))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))
            
            model.add(Dense(units=1024, activation="relu",kernel_initializer=initializer, name="third"))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))

            model.add(Dense(units=1024, activation="relu",kernel_initializer=initializer, name="fourth"))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))

            model.add(Dense(units=256, activation="relu",kernel_initializer=initializer,name="fifth"))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))

               
            model.add(Dense(units=8, activation="softmax"))

elif(args.MODEL == 7): # 4 hidden layers but NOT seccuentially decreasing their size by half and dropout at the end
            model = Sequential()
            model.add(
                Reshape((args.IMG_SIZE * args.IMG_SIZE * 3,), input_shape=(args.IMG_SIZE, args.IMG_SIZE, 3), name='first',
                        dtype='float32'))
            initializer = tf.keras.initializers.GlorotNormal() #default
                    
            model.add(Dense(units=4096, activation="relu",kernel_initializer=initializer, name="second"))
            model.add(BatchNormalization())
                 
            model.add(Dense(units=1024, activation="relu",kernel_initializer=initializer, name="third"))
            model.add(BatchNormalization())

            model.add(Dense(units=1024, activation="relu",kernel_initializer=initializer, name="fourth"))
            model.add(BatchNormalization())

            model.add(Dense(units=256, activation="relu",kernel_initializer=initializer,name="fifth"))
            #model.add(Dense(units=512, activation="relu",kernel_initializer=initializer,name="fifth"))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))

               
            model.add(Dense(units=8, activation="softmax"))

           
           
if args.OPTIMIZER == "sgd":
    opt = tf.keras.optimizers.SGD(learning_rate=args.LEARNING_RATE, momentum=args.MOMENTUM,
                                        weight_decay=args.WEIGHT_DECAY)
elif args.OPTIMIZER == "adam":
    opt = tf.keras.optimizers.Adam(learning_rate=args.LEARNING_RATE, weight_decay=args.WEIGHT_DECAY)
elif args.OPTIMIZER == "rmsprop":
    opt = tf.keras.optimizers.RMSprop(learning_rate=args.LEARNING_RATE, weight_decay=args.WEIGHT_DECAY)
else:
    print("ERROR: optimizer " + args.OPTIMIZER + " not supported!\n")
    quit()

model.compile(loss=args.LOSS, optimizer=opt, metrics=["accuracy"])

print(model.summary())
plot_model(model, to_file="images/modelMLP_" + args.experiment_name + ".png", show_shapes=True, show_layer_names=True)

print("Done!\n")

if os.path.exists(args.MODEL_FNAME):
    print("WARNING: model file " + args.MODEL_FNAME + " exists and will be overwritten!\n")

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
    args.DATASET_DIR + "/train",  # this is the target directory
    target_size=(
        args.IMG_SIZE,
        args.IMG_SIZE,
    ),  # all images will be resized to IMG_SIZExIMG_SIZE
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
    args.DATASET_DIR + "/test",
    target_size=(args.IMG_SIZE, args.IMG_SIZE),
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
    steps_per_epoch=1881 // args.BATCH_SIZE,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=807 // args.BATCH_SIZE,
    verbose=1,
    callbacks=[WandbCallback()]
)

print("Done!\n")

print("Saving the model into " + args.MODEL_FNAME + " \n")
model.save_weights(
    args.WEIGHTS_FNAME + "_" + args.experiment_name + ".h5")  # always save your weights after training or during training
model.save(args.MODEL_FNAME + "_" + args.experiment_name + ".h5")
print("Done!\n")

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

# to get the output of a given layer
# crop the model up to a certain layer
model_layer = Model(inputs=model.input, outputs=model.get_layer("second").output)

# get the features from images
directory = args.DATASET_DIR + "/test/coast"
x = np.asarray(Image.open(os.path.join(directory, os.listdir(directory)[0])))
x = np.expand_dims(np.resize(x, (args.IMG_SIZE, args.IMG_SIZE, 3)), axis=0)
print("prediction for image " + os.path.join(directory, os.listdir(directory)[0]))
features = model_layer.predict(x / 255.0)
print(features)
print("Done!")

wandb.finish()
