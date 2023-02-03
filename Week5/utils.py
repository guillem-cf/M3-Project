import wandb

def preprocess()

 datagen = ImageDataGenerator(
        featurewise_center=True,
        samplewise_center=False,
        featurewise_std_normalization=True,
        samplewise_std_normalization=False,
        preprocessing_function=,
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