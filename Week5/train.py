import matplotlib
import numpy as np
import wandb
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
import tensorflow_addons as tfa
from wandb.keras import WandbCallback
from model import MyModel
from utils import save_plots, get_data_train, get_data_validation, get_data_test, get_optimizer

import tensorflow as tf

matplotlib.use("Agg")

print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# class LearningRateScheduler(WandbCallback):
#     def on_epoch_end(self, epoch, logs={}):
#         lr = self.model.optimizer.lr
#         wandb.log({"lr": tf.keras.backend.get_value(lr)})


def train(args):
    wandb.init(project=args.experiment_name)
    
    tf.random.set_seed(42)
    np.random.seed(42)

    # Print wandb.config
    print(wandb.config)

    args.experiment_name = wandb.config.experiment_name
    
    model = MyModel(name=wandb.config.MODEL, 
                    img_dim = wandb.config.IMG_WIDTH,
                    num_blocks = wandb.config.num_blocks,
                    second_layer = wandb.config.second_layer,
                    third_layer = wandb.config.third_layer,
                    num_denses = wandb.config.num_denses,
                    dim_dense = wandb.config.dim_dense,
                    filters1 = wandb.config.filters1,
                    filters2=wandb.config.filters2, 
                    batch_norm = wandb.config.batch_norm,
                    dropout = wandb.config.dropout,
                    dropout_range = wandb.config.dropout_range,
                    kernel_size=wandb.config.kernel_size,
                    strides=wandb.config.strides, 
                    non_linearities=wandb.config.non_linearities,
                    initializer = wandb.config.initializer,
                    pool_size=wandb.config.pool_size)

    plot_model(model, to_file='./images/model_' + wandb.config.experiment_name + '.png',
               show_shapes=True, show_layer_names=True)

    wandb.config.update({"num_params": model.count_params()})
    model.summary()

    print("Number of parameters: ", wandb.config.num_params)


    # defining the early stop criteria
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=20, mode="auto", min_lr=1e-8)
    # saving the best model based on val_loss
    mc1 = ModelCheckpoint('./checkpoint/best_' + wandb.config.experiment_name + '_model_checkpoint' + '.h5',
                          monitor='val_loss', mode='min', save_best_only=True)
    mc2 = ModelCheckpoint('./checkpoint/best_' + wandb.config.experiment_name + '_model_checkpoint' + '.h5',
                          monitor='val_accuracy', mode='max', save_best_only=True)
    # if wandb.config.CLR_LEARNING_RATE:
    #     clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=wandb.config.INIT_LR,
    #                                                 maximal_learning_rate=wandb.config.MAX_LR,
    #                                                 scale_fn=lambda x: 1. / (2. ** (x - 1)),
    #                                                 step_size = 2 * (int(400 // wandb.config.BATCH_SIZE) + 1))
    #     optimizer = tf.keras.optimizers.Adam(learning_rate=clr)
    # else:
    optimizer = get_optimizer(wandb.config.OPTIMIZER)

    model.compile(loss="categorical_crossentropy", optimizer=optimizer,
                  metrics=["accuracy"])
    
    if wandb.config.CALLBACKS:
        wandb_callback = WandbCallback(input_type="images", labels=["coast", "forest", "highway", "inside_city", "mountain", "Opencountry", "street", "tallbuilding"],
                                   output_type="label", training_data=get_data_train(), validation_data=get_data_validation(), log_weights=True, log_gradients=True, log_evaluation=True, log_batch_frequency=10)
    else:
        wandb_callback = WandbCallback()
        
    history = model.fit(
        get_data_train(),
        steps_per_epoch=(int(400 // wandb.config.BATCH_SIZE) + 1),
        epochs=wandb.config.EPOCHS,
        validation_data=get_data_validation(),
        validation_steps=(int(wandb.config.VALIDATION_SAMPLES // wandb.config.BATCH_SIZE) + 1),
        callbacks=[wandb_callback, mc1, mc2, es, reduce_lr], 
        workers=24
    )
    result = model.evaluate(get_data_test())
    print(result)
    print(history.history.keys())
    save_plots(history, args)
