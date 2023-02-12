import wandb
import tensorflow as tf


class WandbCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        wandb.log(logs, step=epoch)

        best_val_accuracy = wandb.run.summary.get("Best Validation Accuracy", 0.0)
        if logs.get("val_accuracy") > best_val_accuracy:
            wandb.run.summary["Best Validation Accuracy"] = logs.get("val_accuracy")

        best_val_loss = wandb.run.summary.get("Best Validation Loss", float("inf"))
        if logs.get("val_loss") < best_val_loss:
            wandb.run.summary["Best Validation Loss"] = logs.get("val_loss")
