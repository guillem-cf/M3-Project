import matplotlib
import tensorflow as tf
import wandb
from tensorflow.keras import layers
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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
    'method': 'random',
    'name': 'Task4_Task5',
    'metric': {'goal': 'maximize', 'name': 'val_accuracy'},
    'parameters':
        {
            'experiment_name': {'value': args.experiment_name},
            'MODEL_FNAME': {'value': args.MODEL_FNAME},
            'DATASET_DIR': {'value': args.DATASET_DIR},
            'LEARNING_RATE': {'values': [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]},
            'EPOCHS': {'value': 300},
            'BATCH_SIZE': {'values': [10, 32, 64, 128]},
            'OPTIMIZER': {'values': ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']},
            'MOMENTUM': {'values': [0.0, 0.2, 0.4, 0.5, .6, 0.8, 0.9]},
            'LOSS': {'value': args.LOSS},
            'IMG_WIDTH': {'value': args.IMG_WIDTH},
            'IMG_HEIGHT': {'value': args.IMG_HEIGHT},
            'DROPOUT': {'values': [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9]},
            'WEIGHT_DECAY': {'values': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]},
            'VALIDATION_SAMPLES': {'value': args.VALIDATION_SAMPLES},
            'BATCH_NORM_ACTIVE': {'values': [True, False]},
            'data_augmentation_HF': {'values': True},  # [True, False]},
            'data_augmentation_R': {'value': 0},
            # POSAR ELS VALORS QUE ENS SURTIN DEL DATA AUGMENTATION # [0, 20]},#{'max': 20, 'min': 0, 'type': 'int'},
            'data_augmentation_Z': {'value': 0.2},  # [0, 0.2]},#{'max': 0.20, 'min': 0.0, 'type': 'double'},
            'data_augmentation_W': {'value': 0},  # [0, 0.2]},#{'max': 0.20, 'min': 0.0, 'type': 'double'},
            'data_augmentation_H': {'value': 0},  # [0, 0.2]},#{'max': 0.20, 'min': 0.0, 'type': 'double'},
            'data_augmentation_S': {'value': 0},  # [0, 0.2]} #{'max': 0.20, 'min': 0.0, 'type': 'double'}
        }
}

sweep_id = wandb.sweep(sweep=sweep_config, project="M3_W4")


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

    base_model = DenseNet121(include_top=False, weights="imagenet", input_shape=(args.IMG_WIDTH, args.IMG_HEIGHT, 3))
    base_model.trainable = True

    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    # We choose the model 1
    x = base_model.get_layer('pool4_conv').output  # -1 block + -1 transient
    if wandb.config.BATCH_NORM_ACTIVE:
        x = BatchNormalization()(x)

    x = GlobalAveragePooling2D()(x)
    x = Dropout(wandb.config.DROPOUT)(x)
    x = Dense(8, activation='softmax', name='predictionsProf')(x)

    model = Model(inputs=base_model.input, outputs=x)
    # defining the early stop criteria
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6)
    # saving the best model based on val_loss
    mc1 = ModelCheckpoint('./checkpoint/best_' + args.experiment_name + '_model_checkpoint' + '.h5',
                          monitor='val_loss', mode='min', save_best_only=True)
    mc2 = ModelCheckpoint('./checkpoint/best_' + args.experiment_name + '_model_checkpoint' + '.h5',
                          monitor='val_accuracy', mode='max', save_best_only=True)

    model.compile(loss="categorical_crossentropy", optimizer=get_optimizer(wandb.config.OPTIMIZER),
                  metrics=["accuracy"])

    # preprocessing_function=preprocess_input,

    history = model.fit(
        train_generator,
        steps_per_epoch=(int(400 // wandb.config.BATCH_SIZE) + 1),
        epochs=wandb.config.EPOCHS,
        validation_data=validation_generator,
        validation_steps=(int(wandb.config.VALIDATION_SAMPLES // wandb.config.BATCH_SIZE) + 1),
        callbacks=[WandbCallback(), mc1, mc2, es, reduce_lr],
        use_multiprocessing=True,
        workers=16
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


wandb.agent(sweep_id, function=train, count=30)
