# python main.py --config config/best_model.yaml

import argparse
import functools
import wandb
import yaml

from train import train


def main():
    parser = argparse.ArgumentParser(
        description="MIT", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--experiment_name", type=str, help="Experiment name", default='baseline')
    parser.add_argument("--config", type=str, help="Config file", required=True)
    # parser.add_argument("--MODEL", type=str, help="Indicate the model to use", default="mlp")
    # parser.add_argument("--DATASET_DIR", type=str, help="Dataset path",
    #                     default="/ghome/group07/M3-Project-new/Week4/MIT_small_train_1")
    # parser.add_argument("--VALIDATION_SAMPLES", type=int, help="Indicate Validation Samples", default=807)
    # parser.add_argument("--CALLBACKS", type=bool, help="Image callbacks", default=False)

    # parser.add_argument("--IMG_WIDTH", type=int, help="Indicate Image Size", default=64)  # abans 224
    # parser.add_argument("--IMG_HEIGHT", type=int, help="Indicate Image Size", default=64)  # abans 224

    # parser.add_argument("--BATCH_SIZE", type=int, help="Indicate Batch Size", default=32)
    # parser.add_argument("--EPOCHS", type=int, help="Indicate Epochs", default=300)
    # parser.add_argument("--LEARNING_RATE", type=float, help="Indicate Learning Rate", default=0.001)
    # parser.add_argument("--MOMENTUM", type=float, help="Indicate Momentum", default=0.9)
    # parser.add_argument("--WEIGHT_DECAY", type=float, help="Indicate Weight Decay", default=0.0001)
    # parser.add_argument("--OPTIMIZER", type=str, help="Indicate Optimizer", default="adam")
    # parser.add_argument("--LOSS", type=str, help="Indicate Loss", default="categorical_crossentropy")


    # parser.add_argument("--num_blocks", type=int, help="Number of blocks", default=4)
    # parser.add_argument("--second_layer", type=bool, help="Second layer", default=True)
    # parser.add_argument("--num_denses", type=int, help="Number of denses", default=3)
    # parser.add_argument("--filters1", type=int, help="Filters 1", default=64)
    # parser.add_argument("--filters2", type=int, help="Filters 2", default=64)
    # parser.add_argument("--batch_norm", type=bool, help="Batch Normalization", default=True)
    # parser.add_argument("--dropout", type=bool, help="Dropout", default=True)
    # parser.add_argument("--dropout_range", type=float, help="Dropout Range", default=0.3)
    # parser.add_argument('--kernel_size', type=int, help='kernel_size', default=3)
    # parser.add_argument('--strides', type=int, help='stride', default=1)
    # parser.add_argument('--initializer', type=str, help='initializer', default='HeUniform')
    # parser.add_argument("--non_linearities", type=str, help="Indicate Non Linearity", default="leaky_relu")
    # parser.add_argument('--pool_size', type=int, help='pool size', default=2)

    # parser.add_argument('--hardcore_data_augmentation', type=bool, help='hardcore_data_augmentation', default=True)
    # parser.add_argument("--horizontal_flip", type=bool, help="Horizontal Flip", default=True)
    # parser.add_argument("--vertical_flip", type=bool, help="Vertical Flip", default=False)
    # parser.add_argument("--rotation", type=int, help="Rotation", default=20)
    # parser.add_argument("--width_shift", type=float, help="Width Shift", default=0.02)
    # parser.add_argument("--height_shift", type=float, help="Height Shift", default=0.02)
    # parser.add_argument("--shear_range", type=float, help="Shear Range", default=0.2)
    # parser.add_argument("--zoom_range", type=float, help="Zoom Range", default=0.2)
    # parser.add_argument("--brightness_range", nargs='+', type=float, help="Brightness Range", default=[0.2, 0.8])
    # parser.add_argument("--reescaling", type=float, help="Reescaling", default=1.0 / 255.0)
    args = parser.parse_args()

    # sweep_config = {
    #     'method': 'random',
    #     'name': 'Sweep_' + args.experiment_name,
    #     'metric': {'goal': 'maximize', 'name': 'val_accuracy'},
    #     'parameters':
    #         {
    #             'experiment_name': {'value': args.experiment_name},
    #             'MODEL': {'value': args.MODEL},
    #             'DATASET_DIR': {'value': args.DATASET_DIR},
    #             'VALIDATION_SAMPLES': {'value': args.VALIDATION_SAMPLES},
    #             'CALLBACKS': {'value': args.CALLBACKS},

    #             'IMG_WIDTH': {'value': args.IMG_WIDTH},
    #             'IMG_HEIGHT': {'value': args.IMG_HEIGHT},

    #             'BATCH_SIZE': {'value': args.BATCH_SIZE},
    #             'EPOCHS': {'value': args.EPOCHS},
    #             'LEARNING_RATE': {'value': args.LEARNING_RATE},
    #             'MOMENTUM': {'value': args.MOMENTUM},  # 'values': [0.9, 0.99, 0.999]
    #             'WEIGHT_DECAY': {'value': args.WEIGHT_DECAY},
    #             'OPTIMIZER': {'value': args.OPTIMIZER},
    #             'LOSS': {'value': args.LOSS},

    #             'num_blocks': {'value': args.num_blocks},
    #             'second_layer': {'value': args.second_layer},
    #             'num_denses': {'value': args.num_denses},
    #             'filters1': {'value': args.filters1},
    #             'filters2': {'value': args.filters2},
    #             'batch_norm': {'value': args.batch_norm},
    #             'dropout': {'value': args.dropout},
    #             'dropout_range': {'value': args.dropout_range},
    #             'kernel_size': {'value': args.kernel_size},
    #             'strides': {'value': args.strides},
    #             'initializer': {'value': args.initializer},
    #             'non_linearities': {'value': args.non_linearities},
    #             'pool_size': {'value': args.pool_size},

    #             'hardcore_data_augmentation': {'value': args.hardcore_data_augmentation},
    #             'horizontal_flip': {'value': args.horizontal_flip},
    #             'vertical_flip': {'value': args.vertical_flip},
    #             'rotation': {'value': args.rotation},
    #             # {'values': [0, 20]},  # {'max': 0.20, 'min': 0.0, 'type': 'double'},
    #             'width_shift': {'value': args.width_shift},
    #             # {'values': [0, 0.2]},  # {'max': 0.20, 'min': 0.0, 'type': 'double'},
    #             'height_shift': {'value': args.height_shift},
    #             # {'values': [0, 0.2]}  # {'max': 0.20, 'min': 0.0, 'type': 'double'}
    #             'shear_range': {'value': args.shear_range},
    #             # {'values': [0, 0.2]},  # {'max': 0.20, 'min': 0.0, 'type': 'double'},
    #             'zoom_range': {'value': args.zoom_range},
    #             # {'values': [0, 0.2]},  # {'max': 20, 'min': 0, 'type': 'int'},
    #             'brightness_range': {'value': args.brightness_range}
    #             # {'values': [0, 0.2]},  # {'max': 0.20, 'min': 0.0, 'type': 'double'}
    #         }
    # }
    # Read the sweep_config from a yaml


    with open(args.config, "r") as f:
        sweep_config = yaml.safe_load(f)

    sweep_id = wandb.sweep(sweep=sweep_config, project="M3_W5")
    wandb.agent(sweep_id, function=functools.partial(train, args))

if __name__ == "__main__":
    main()
    wandb.finish()
