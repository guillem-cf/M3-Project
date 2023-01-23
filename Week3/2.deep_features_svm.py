import argparse
import pickle as pkl
import pandas as pd


import matplotlib
import tensorflow as tf
import wandb
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Flatten, Dense, Reshape
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
from wandb.keras import WandbCallback

from utils import *
from BOVW import BoVW, svm, plotROC_BWVW

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from sklearn.preprocessing import normalize,  LabelBinarizer, StandardScaler

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

parser = argparse.ArgumentParser(description="MIT", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--DATASET_DIR", type=str, help="Dataset path", default="./MIT_split")
parser.add_argument("--PATCHES_DIR", type=str, help="Patches path", default="./MIT_split_patches")
parser.add_argument("--MODEL_FNAME", type=str, default="./best_models/mlp_hidLayers_4_big2.h5", help="Model path")
parser.add_argument("--OUTPUT_LAYER", type=str, default="second", help="Layer output")
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
parser.add_argument("--experiment_name", type=str, help="Experiment name", default="baseline")
parser.add_argument("--KERNEL", type=str, help="Experiment name", default="RBF")
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
)

# wandb.init(
#     project="M3",
#     config=config,
#     name=args.experiment_name,
# )

PATCHES_DIR = args.PATCHES_DIR + str(args.PATCH_SIZE)

if not os.path.exists(args.DATASET_DIR):
    print("ERROR: dataset directory " + args.DATASET_DIR + " does not exist!\n")
    quit()

# Load the dataset
train_images_filenames = pkl.load(open("MIT_split/train_images_filenames.dat", "rb"))
test_images_filenames = pkl.load(open("MIT_split/test_images_filenames.dat", "rb"))
train_images_filenames = [n[16:] for n in train_images_filenames]
test_images_filenames = [n[16:] for n in test_images_filenames]
train_labels = pkl.load(open("MIT_split/train_labels.dat", "rb"))
test_labels = pkl.load(open("MIT_split/test_labels.dat", "rb"))

# Load the model
model = load_model(args.MODEL_FNAME)
if args.OUTPUT_LAYER == "second":
    model_layer = Model(inputs=model.input, outputs=model.get_layer("second").output)
elif args.OUTPUT_LAYER == "third":
    model_layer = Model(inputs=model.input, outputs=model.get_layer("third").output)
elif args.OUTPUT_LAYER == "fourth":
    model_layer = Model(inputs=model.input, outputs=model.get_layer("fourth").output)
elif args.OUTPUT_LAYER == "fifth":
    model_layer = Model(inputs=model.input, outputs=model.get_layer("fifht").output)
model.summary()

PATCH_SIZE = model.layers[0].input.shape[1:3]
NUM_PATCHES = (args.IMG_SIZE//PATCH_SIZE.as_list()[0])**2


def get_features(image_filenames, model_layer):
    # Get features for all images
    features = []
    i = 0
    for filename in image_filenames:
        x = np.asarray(Image.open(filename))
        x = np.expand_dims(np.resize(x, (args.IMG_SIZE, args.IMG_SIZE, 3)), axis=0)
        feature = model_layer.predict(x / 255.0)
        features.append(feature[0])
        i += 1
        # if(i==300):
        #     break
    return np.array(features)

# Get features for all images in the training set
train_features = get_features(train_images_filenames, model_layer)

# Get features for all images in the test set
test_features = get_features(test_images_filenames, model_layer)


print(f'Train features shape: {train_features.shape}')
print(f'Train labels shape: {len(train_labels)}')
print(f'Test features shape: {test_features.shape}')
print(f'Test labels shape: {len(test_labels)}')

scores = pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=False)

classifier, scores, accuracy = svm(train_features, train_labels, test_features, test_labels, args.KERNEL)

print(scores)



# Save pandas dataframe scores in a .jpg file
# path = "deep_features_svm/accuracy_" + args.experiment_name + ".txt"
# np.savetxt(path, scores.values, fmt='%d')
path = "deep_features_svm/accuracy_" + args.experiment_name + ".csv"
scores.to_csv(path, sep='\t', encoding='utf-8')


print(f'Accuracy = {accuracy}')


labels = [
    "Opencountry",
    "coast",
    "forest",
    "mountain",
    "highway",
    "tallbuilding",
    "street",
    "inside_city",
]

# plotROC_BWVW(
#     train_labels,
#     test_labels,
#     train_features,
#     test_features,
#     labels,
#     classifier,
# )

