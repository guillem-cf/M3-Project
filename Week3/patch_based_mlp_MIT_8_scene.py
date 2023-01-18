from __future__ import print_function

from keras.layers import Dense, Reshape
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import argparse

from utils import *


parser = argparse.ArgumentParser(description='MIT',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--DATASET_DIR', type=str, help='Dataset path', default='/home/mcv/datasets/MIT_split')
parser.add_argument('--PATCHES_DIR', type=str, help='Patches path', default='/home/group10/m3/data/MIT_split_patches')
parser.add_argument('--MODEL_FNAME', type=str, default='/home/group10/m3/patch_based_mlp.h5', help='Model path')
parser.add_argument('--PATCH_SIZE', type=int, help='Indicate Patch Size', default=64)
parser.add_argument("--BATCH_SIZE", type=int, help="Indicate Batch Size", default=16)
args = parser.parse_args()

# user defined variables
"""
PATCH_SIZE = 64
BATCH_SIZE = 16
DATASET_DIR = '/home/mcv/datasets/MIT_split'
PATCHES_DIR = '/home/group10/m3/data/MIT_split_patches' + str(PATCH_SIZE)
MODEL_FNAME = '/home/group10/m3/patch_based_mlp.h5'
"""
PATCHES_DIR = args.PATCHES_DIR + str(args.PATCH_SIZE)

def build_mlp(input_size=args.PATCH_SIZE, phase='TRAIN'):
    model = Sequential()
    model.add(Reshape((input_size * input_size * 3,), input_shape=(input_size, input_size, 3)))
    model.add(Dense(units=2048, activation='relu'))
    # model.add(Dense(units=1024, activation='relu'))
    if phase == 'TEST':
        model.add(
            Dense(units=8, activation='linear'))  # In test phase we softmax the average output over the image patches
    else:
        model.add(Dense(units=8, activation='softmax'))
    return model


if not os.path.exists(args.DATASET_DIR):
    print('ERROR: dataset directory ' + args.DATASET_DIR + ' do not exists!\n')
    quit()
if not os.path.exists(args.PATCHES_DIR):
    print('WARNING: patches dataset directory ' + PATCHES_DIR + ' does not exist!\n')
    print('Creating image patches dataset into ' + PATCHES_DIR + '\n')
    generate_image_patches_db(args.DATASET_DIR, PATCHES_DIR, patch_size=args.PATCH_SIZE)
    print('Done!\n')

print('Building MLP model...\n')

model = build_mlp(input_size=args.PATCH_SIZE)

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

print(model.summary())

print('Done!\n')

train = True
if not os.path.exists(args.MODEL_FNAME) or train:
    print('WARNING: model file ' + args.MODEL_FNAME + ' do not exists!\n')
    print('Start training...\n')
    # this is the dataset configuration we will use for training
    # only rescaling
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=True)

    # this is the dataset configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # this is a generator that will read pictures found in
    # subfolers of 'data/train', and indefinitely generate
    # batches of augmented image data
    train_generator = train_datagen.flow_from_directory(
        PATCHES_DIR + '/train',  # this is the target directory
        target_size=(args.PATCH_SIZE, args.PATCH_SIZE),  # all images will be resized to PATCH_SIZExPATCH_SIZE
        batch_size=args.BATCH_SIZE,
        classes=['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding'],
        class_mode='categorical')  # since we use binary_crossentropy loss, we need categorical labels

    # this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
        PATCHES_DIR + '/test',
        target_size=(args.PATCH_SIZE, args.PATCH_SIZE),
        batch_size=args.BATCH_SIZE,
        classes=['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding'],
        class_mode='categorical')

    model.fit(
        train_generator,
        steps_per_epoch=18810 // args.BATCH_SIZE,
        epochs=150,
        validation_data=validation_generator,
        validation_steps=8070 // args.BATCH_SIZE)

    print('Done!\n')
    print('Saving the model into ' + args.MODEL_FNAME + ' \n')
    model.save_weights(args.MODEL_FNAME)  # always save your weights after training or during training
    print('Done!\n')

print('Building MLP model for testing...\n')

model = build_mlp(input_size=args.PATCH_SIZE, phase='TEST')
print(model.summary())

print('Done!\n')

print('Loading weights from ' + args.MODEL_FNAME + ' ...\n')
print('\n')

model.load_weights(args.MODEL_FNAME)

print('Done!\n')

print('Start evaluation ...\n')

directory = args.DATASET_DIR + '/test'
classes = {'coast': 0, 'forest': 1, 'highway': 2, 'inside_city': 3, 'mountain': 4, 'Opencountry': 5, 'street': 6,
           'tallbuilding': 7}
correct = 0.
total = 807
count = 0

for class_dir in os.listdir(directory):
    cls = classes[class_dir]
    for imname in os.listdir(os.path.join(directory, class_dir)):
        im = Image.open(os.path.join(directory, class_dir, imname))
        patches = image.extract_patches_2d(np.array(im), (args.PATCH_SIZE, args.PATCH_SIZE), max_patches=1)
        out = model.predict(patches / 255.)
        predicted_cls = np.argmax(softmax(np.mean(out, axis=0)))
        if predicted_cls == cls:
            correct += 1
        count += 1
        print('Evaluated images: ' + str(count) + ' / ' + str(total), end='\r')

print('Done!\n')
print('Test Acc. = ' + str(correct / total) + '\n')
