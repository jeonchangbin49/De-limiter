import os
import argparse
import random

import torch

from train_ddp import train
from utils import get_config


def main():
    parser = argparse.ArgumentParser(description="Trainer")

    # Put every argumnet in './configs/yymmdd_architecture_number.yaml' and load it.
    parser.add_argument(
        "-c",
        "--config",
        default="delimit_6_s",
        type=str,
        help="Name of the setting file.",
    )

    config_args = parser.parse_args()

    args = get_config(config_args.config)

    args.img_check = (
        f"{args.dir_params.output_directory}/img_check/{args.dir_params.exp_name}"
    )
    args.output = (
        f"{args.dir_params.output_directory}/checkpoint/{args.dir_params.exp_name}"
    )

    # Set which devices to use
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(random.randint(0, 1800))

    os.makedirs(args.img_check, exist_ok=True)
    os.makedirs(args.output, exist_ok=True)

    torch.manual_seed(args.sys_params.seed)
    random.seed(args.sys_params.seed)

    print(args)
    train(args)


if __name__ == "__main__":
    main()
