import argparse
import functools
import wandb

from train import train


def main():
    parser = argparse.ArgumentParser(
        description="MIT", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--experiment_name", type=str, help="Experiment name", default="baseline")
    parser.add_argument("--MODEL", type=str, help="Indicate the model to use", default="mlp")
    parser.add_argument("--DATASET_DIR", type=str, help="Dataset path", default="/ghome/group07/M3-Project-new/Week4/MIT_small_train_1")
    parser.add_argument("--VALIDATION_SAMPLES", type=int, help="Indicate Validation Samples", default=807)
    parser.add_argument("--CALLBACKS", type=bool, help="Image callbacks", default=False)

    parser.add_argument("--IMG_WIDTH", type=int, help="Indicate Image Size", default=256)  # abans 224
    parser.add_argument("--IMG_HEIGHT", type=int, help="Indicate Image Size", default=256) # abans 224

    parser.add_argument("--BATCH_SIZE", type=int, help="Indicate Batch Size", default=32)
    parser.add_argument("--EPOCHS", type=int, help="Indicate Epochs", default=200)
    parser.add_argument("--LEARNING_RATE", type=float, help="Indicate Learning Rate", default=0.001)
    parser.add_argument("--MOMENTUM", type=float, help="Indicate Momentum", default=0.9)
    parser.add_argument("--WEIGHT_DECAY", type=float, help="Indicate Weight Decay", default=0.0001)
    parser.add_argument("--OPTIMIZER", type=str, help="Indicate Optimizer", default="adam")
    parser.add_argument("--LOSS", type=str, help="Indicate Loss", default="categorical_crossentropy")

    parser.add_argument('--filters', nargs='+', type=int, help='filters', default=[32, 64, 128])
    parser.add_argument('--kernel_size', nargs='+', type=int, help='kernel_size', default=[5, 3])
    parser.add_argument('--strides', type=int, help='stride', default=1)
    parser.add_argument('--pool_size', type=int, help='pool size', default=2)
    parser.add_argument("--DROPOUT", type=float, help="Indicate Dropout", default=0.2)

    parser.add_argument("--horizontal_flip", type=bool, help="Horizontal Flip", default=False)
    parser.add_argument("--vertical_flip", type=bool, help="Vertical Flip", default=False)
    parser.add_argument("--rotation", type=int, help="Rotation", default=0)
    parser.add_argument("--width_shift", type=float, help="Width Shift", default=0.0)
    parser.add_argument("--height_shift", type=float, help="Height Shift", default=0.0)
    parser.add_argument("--shear_range", type=float, help="Shear Range", default=0.0)
    parser.add_argument("--zoom_range", type=float, help="Zoom Range", default=0.0)
    parser.add_argument("--brightness_range", nargs='+', type=float, help="Brightness Range", default=[0.2,0.8])
    parser.add_argument("--reescaling", type=float, help="Reescaling", default=1.0/255.0)
    
    args = parser.parse_args()


    # wandb.init(project="M3_W5", name=args.experiment_name)
    # wandb.config.update(args)
    # sweep_config = sweep(args)
    sweep_config = {
        'method': 'random',
        'name': 'Sweep_' + args.experiment_name,
        'metric': {'goal': 'maximize', 'name': 'val_accuracy'},
        'parameters':
            {
                'experiment_name': {'value': args.experiment_name},
                'MODEL': {'value': args.MODEL},
                'DATASET_DIR': {'value': args.DATASET_DIR},
                'VALIDATION_SAMPLES': {'value': args.VALIDATION_SAMPLES},
                'CALLBACKS': {'value': args.CALLBACKS},

                'IMG_WIDTH': {'value': args.IMG_WIDTH},
                'IMG_HEIGHT': {'value': args.IMG_HEIGHT},
                
                'BATCH_SIZE': {'value': args.BATCH_SIZE},
                'EPOCHS': {'value': args.EPOCHS},
                'LEARNING_RATE': {'value': args.LEARNING_RATE},
                'MOMENTUM': {'value': args.MOMENTUM},                # 'values': [0.9, 0.99, 0.999]
                'WEIGHT_DECAY': {'value': args.WEIGHT_DECAY},
                'OPTIMIZER': {'value': args.OPTIMIZER},
                'LOSS': {'value': args.LOSS},
                
                'filters': {'value': args.filters},
                'kernel_size': {'value': args.kernel_size},
                'strides': {'value': args.strides},
                'pool_size': {'value': args.pool_size},
                'DROPOUT': {'value': args.DROPOUT},
                
                'horizontal_flip':{'value':args.horizontal_flip},     
                'vertical_flip':{'value':args.vertical_flip},       
                'rotation':{'value':args.rotation},                 # {'values': [0, 20]},  # {'max': 0.20, 'min': 0.0, 'type': 'double'},
                'width_shift':{'value':args.width_shift},           # {'values': [0, 0.2]},  # {'max': 0.20, 'min': 0.0, 'type': 'double'},
                'height_shift':{'value':args.height_shift},         # {'values': [0, 0.2]}  # {'max': 0.20, 'min': 0.0, 'type': 'double'}
                'shear_range':{'value':args.shear_range},           # {'values': [0, 0.2]},  # {'max': 0.20, 'min': 0.0, 'type': 'double'},
                'zoom_range':{'value':args.zoom_range},             # {'values': [0, 0.2]},  # {'max': 20, 'min': 0, 'type': 'int'},  
                'brightness_range':{'value':args.brightness_range}  # {'values': [0, 0.2]},  # {'max': 0.20, 'min': 0.0, 'type': 'double'} 
            }
    }

    sweep_id = wandb.sweep(sweep=sweep_config, project="M3_W5")
    wandb.agent(sweep_id, function=functools.partial(train, args), count=1)

if __name__ == "__main__":
    main()
    wandb.finish()
