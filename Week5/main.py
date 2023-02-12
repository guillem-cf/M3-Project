# python main.py --config best_models/ResNet.yaml

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
    args = parser.parse_args()


    with open(args.config, "r") as f:
        sweep_config = yaml.safe_load(f)

    sweep_id = wandb.sweep(sweep=sweep_config, project="M3_W5")
    wandb.agent(sweep_id, function=functools.partial(train, args))

if __name__ == "__main__":
    main()
    wandb.finish()
