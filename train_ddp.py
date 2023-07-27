import sys
import time

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import wandb

from solver_ddp import Solver


def train(args):
    print("hello")
    solver = Solver()

    ngpus_per_node = int(torch.cuda.device_count() / args.sys_params.n_nodes)
    print(f"use {ngpus_per_node} gpu machine")
    args.sys_params.world_size = ngpus_per_node * args.sys_params.n_nodes
    mp.spawn(worker, nprocs=ngpus_per_node, args=(solver, ngpus_per_node, args))


def worker(gpu, solver, ngpus_per_node, args):
    args.sys_params.rank = args.sys_params.rank * ngpus_per_node + gpu
    dist.init_process_group(
        backend="nccl",
        world_size=args.sys_params.world_size,
        init_method="env://",
        rank=args.sys_params.rank,
    )
    args.gpu = gpu
    args.ngpus_per_node = ngpus_per_node

    solver.set_gpu(args)  # 여기서 resume도 일어남.
    # rank라는 것은 8개의 process들 사이의 우선순위를 의미 함.

    start_epoch = solver.start_epoch

    if args.dir_params.resume:
        start_epoch = start_epoch + 1

    for epoch in range(start_epoch, args.hyperparams.epochs + 1):

        solver.train_sampler.set_epoch(epoch)
        solver.train(args, epoch)

        time.sleep(1)

        solver.multi_validate(args, epoch)

        if solver.stop == True:
            print("Apply Early Stopping")
            if args.wandb_params.use_wandb:
                wandb.finish()
            sys.exit()

    if args.wandb_params.use_wandb:
        wandb.finish()
