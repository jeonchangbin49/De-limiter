import os
import argparse
import random

import torch
from deepspeed.profiling.flops_profiler import get_model_profile

from utils import get_config
from models import load_model_with_args


# def main():
parser = argparse.ArgumentParser(description="Demucs Trainer")

# Put every argumnet in './configs/yymmdd_architecture_number.yaml' and load it.
parser.add_argument(
    "-c", "--config", default="default", type=str, help="Name of the setting file."
)

config_args = parser.parse_args()

args = get_config(config_args.config)
# args = get_config("230402_delimit_convtasnet_35")
print(args)

with torch.cuda.device(0):
    model = load_model_with_args(args)
    # print(model)
    batch_size = 1
    flops, macs, params = get_model_profile(
        model=model,  # model
        input_shape=(batch_size, 2, 44100 * 60),  # input shape to the model. If specified, the model takes a tensor with this shape as the only positional argument.
        args=[],  # list of positional arguments to the model.
        kwargs={},  # dictionary of keyword arguments to the model.
        print_profile=True,  # prints the model graph with the measured profile attached to each module
        detailed=True,  # print the detailed profile
        module_depth=-1,  # depth into the nested modules, with -1 being the inner most modules
        top_modules=1,  # the number of top modules to print aggregated profile
        warm_up=1,  # the number of warm-ups before measuring the time of each module
        as_string=True,  # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
        output_file=None,  # path to the output file. If None, the profiler prints to stdout.
        ignore_modules=None,
    )  # the list of modules to ignore in the profiling
    print(args.dir_params.exp_name)
    print('flops: ', flops)
    print('macs: ', macs)
    print('params: ', params)


# In [7]: flops
# Out[7]: '8899.46 G'

# In [8]: macs
# Out[8]: '4448.16 GMACs'

# In [9]: params
# Out[9]: '2.35 M'

