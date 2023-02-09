import matplotlib
import tensorflow as tf
import wandb
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
from wandb.keras import WandbCallback
from model import MyModel
from utils import save_plots, get_data_train, get_data_validation, get_data_test, get_optimizer

matplotlib.use("Agg")

print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def train(args):
    wandb.init(project=args.experiment_name)
    
    # Print wandb.config
    print(wandb.config)

    model = MyModel(name=args.MODEL, filters=wandb.config.filters, kernel_size=wandb.config.kernel_size, strides=wandb.config.strides,
                    pool_size=wandb.config.pool_size,
                    dropout_rate=wandb.config.DROPOUT, non_linearities=wandb.config.NON_LINEARITY)
    plot_model(model, to_file='./images/model_' + wandb.config.experiment_name + '.png',
               show_shapes=True, show_layer_names=True)
    model.summary()

    # defining the early stop criteria
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=20, min_lr=1e-8)
    # saving the best model based on val_loss
    mc1 = ModelCheckpoint('./checkpoint/best_' + wandb.config.experiment_name + '_model_checkpoint' + '.h5',
                          monitor='val_loss', mode='min', save_best_only=True)
    mc2 = ModelCheckpoint('./checkpoint/best_' + wandb.config.experiment_name + '_model_checkpoint' + '.h5',
                          monitor='val_accuracy', mode='max', save_best_only=True)
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
        callbacks=[wandb_callback, mc1, mc2, es, reduce_lr], workers=24
    )
    result = model.evaluate(get_data_test())
    print(result)
    print(history.history.keys())
    save_plots(history, args)
