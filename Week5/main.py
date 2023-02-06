import argparse
import wandb
from utils import sweep
from baseline import train
def main():
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
    parser.add_argument("--OPTIMIZER", type=str, help="Indicate Optimizer", default="adam")
    parser.add_argument("--LOSS", type=str, help="Indicate Loss", default="categorical_crossentropy")
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
    wandb.init(project=args.experiment_name)
    sweep_config = sweep(args)
    sweep_id = wandb.sweep(sweep=sweep_config, project="M3_W5")
    wandb.agent(sweep_id, function=train)

if __name__ == "__main__":
    main()
