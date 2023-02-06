import tensorflow as tf
import wandb
from wandb.keras import WandbCallback
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
from model import MyModel
from utils import save_plots, get_data_train, get_data_validation, get_data_test, sweep
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def train(args):
    model = MyModel(name=args.experiment_name, filters=32, kernel_size=3, strides=1, pool_size=2,
                    dropout_rate=wandb.config.DROPOUT, non_linearities="relu")
    
    plot_model(model, to_file='./images/model_'+ args.experiment_name + '.png',
               show_shapes=True, show_layer_names=True)

    # defining the early stop criteria
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6)
    # saving the best model based on val_loss
    mc1 = ModelCheckpoint('./checkpoint/best_' + args.experiment_name + '_model_checkpoint' + '.h5',
                          monitor='val_loss', mode='min', save_best_only=True)
    mc2 = ModelCheckpoint('./checkpoint/best_' + args.experiment_name + '_model_checkpoint' + '.h5',
                          monitor='val_accuracy', mode='max', save_best_only=True)

    model.compile(loss="categorical_crossentropy", optimizer=wandb.config.OPTIMIZER,
                  metrics=["accuracy"])

    history = model.fit(
        get_data_train(),
        steps_per_epoch=(int(400 // wandb.config.BATCH_SIZE) + 1),
        epochs=wandb.config.EPOCHS,
        validation_data=get_data_validation(),
        validation_steps=(int(wandb.config.VALIDATION_SAMPLES // wandb.config.BATCH_SIZE) + 1),
        callbacks=[WandbCallback(), mc1, mc2, es, reduce_lr],
    )
    result = model.evaluate(get_data_test())
    print(result)
    print(history.history.keys())
    save_plots(history, args)
