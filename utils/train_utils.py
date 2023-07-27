import argparse

import yaml
from dotmap import DotMap
import numpy as np


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_config(config_name="default"):

    with open(f"./configs/{config_name}.yaml", "r") as f:

        config = yaml.load(f, Loader=yaml.FullLoader)
    config = DotMap(config)
    return config
