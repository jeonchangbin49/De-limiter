import time
import json

import torch
import torch.nn as nn
import wandb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from asteroid.losses import (
    pairwise_neg_sisdr,
    PairwiseNegSDR,
)
from einops import rearrange, reduce
from ema_pytorch import EMA

from models import load_model_with_args
import utils
from dataloader import (
    MusdbTrainDataset,
    MusdbValidDataset,
    DelimitTrainDataset,
    DelimitValidDataset,
    OzoneTrainDataset,
    OzoneValidDataset,
    aug_from_str,
    SingleTrackSet,
)


class Solver(object):
    def __init__(self):
        pass

    def set_gpu(self, args):

        if args.wandb_params.use_wandb and args.gpu == 0:
            if args.wandb_params.sweep:
                wandb.init(
                    entity=args.wandb_params.entity,
                    project=args.wandb_params.project,
                    config=args,
                    resume=True
                    if args.dir_params.resume != None and args.gpu == 0
                    else False,
                )
            else:
                wandb.init(
                    entity=args.wandb_params.entity,
                    project=args.wandb_params.project,
                    name=f"{args.dir_params.exp_name}",
                    config=args,
                    resume="must"
                    if args.dir_params.resume != None
                    and not args.dir_params.continual_train
                    else False,
                    id=args.wandb_params.rerun_id
                    if args.wandb_params.rerun_id
                    else None,
                    settings=wandb.Settings(start_method="fork"),
                )

        ###################### Define Models ######################
        self.model = load_model_with_args(args)

        trainable_params = []
        trainable_params = trainable_params + list(self.model.parameters())

        if args.hyperparams.optimizer == "sgd":
            print("Use SGD optimizer.")
            self.optimizer = torch.optim.SGD(
                params=trainable_params,
                lr=args.hyperparams.lr,
                momentum=0.9,
                weight_decay=args.hyperparams.weight_decay,
            )
        elif args.hyperparams.optimizer == "adamw":
            print("Use AdamW optimizer.")
            self.optimizer = torch.optim.AdamW(
                params=trainable_params,
                lr=args.hyperparams.lr,
                betas=(0.9, 0.999),
                amsgrad=False,
                weight_decay=args.hyperparams.weight_decay,
            )
        elif args.hyperparams.optimizer == "radam":
            print("Use RAdam optimizer.")
            self.optimizer = torch.optim.RAdam(
                params=trainable_params,
                lr=args.hyperparams.lr,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=args.hyperparams.weight_decay,
            )
        elif args.hyperparams.optimizer == "adam":
            print("Use Adam optimizer.")
            self.optimizer = torch.optim.Adam(
                params=trainable_params,
                lr=args.hyperparams.lr,
                betas=(0.9, 0.999),
                weight_decay=args.hyperparams.weight_decay,
            )
        else:
            print("no optimizer loaded")
            raise NotImplementedError

        if args.hyperparams.lr_scheduler == "step_lr":
            if args.model_loss_params.architecture == "umx":
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode="min",
                    factor=args.hyperparams.lr_decay_gamma,
                    patience=args.hyperparams.lr_decay_patience,
                    cooldown=10,
                    verbose=True,
                )
            else:
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode="min",
                    factor=args.hyperparams.lr_decay_gamma,
                    patience=args.hyperparams.lr_decay_patience,
                    cooldown=0,
                    min_lr=5e-5,
                    verbose=True,
                )
        elif args.hyperparams.lr_scheduler == "cos_warmup":
            self.scheduler = utils.CosineAnnealingWarmUpRestarts(
                self.optimizer,
                T_0=40,
                T_mult=1,
                eta_max=args.hyperparams.lr,
                T_up=10,
                gamma=0.5,
            )

        torch.cuda.set_device(args.gpu)

        self.model = self.model.to(f"cuda:{args.gpu}")

        ############################################################
        # Define Losses
        self.criterion = {}

        self.criterion["l1"] = nn.L1Loss().to(args.gpu)
        self.criterion["mse"] = nn.MSELoss().to(args.gpu)
        self.criterion["si_sdr"] = pairwise_neg_sisdr.to(args.gpu)
        self.criterion["snr"] = PairwiseNegSDR("snr").to(args.gpu)
        self.criterion["bcewithlogits"] = nn.BCEWithLogitsLoss().to(args.gpu)
        self.criterion["bce"] = nn.BCELoss().to(args.gpu)
        self.criterion["kl"] = nn.KLDivLoss(log_target=True).to(args.gpu)

        print("Loss functions we use in this training:")
        print(args.model_loss_params.train_loss_func)

        # Early stopping utils
        self.es = utils.EarlyStopping(patience=args.hyperparams.patience)
        self.stop = False

        if args.wandb_params.use_wandb and args.gpu == 0:
            wandb.watch(self.model, log="all")

        self.start_epoch = 1
        self.train_losses = []
        self.valid_losses = []
        self.train_times = []
        self.best_epoch = 0

        if args.dir_params.resume and not args.hyperparams.ema:
            self.resume(args)

        # Distribute models to machine
        self.model = DDP(
            self.model,
            device_ids=[args.gpu],
            output_device=args.gpu,
            find_unused_parameters=True,
        )

        if args.hyperparams.ema:
            self.model_ema = EMA(
                self.model,
                beta=0.999,
                update_after_step=100,
                update_every=10,
            )

        if args.resume and args.hyperparams.ema:
            self.resume(args)

        ###################### Define data pipeline ######################
        args.hyperparams.batch_size = int(
            args.hyperparams.batch_size / args.ngpus_per_node
        )
        self.mp_context = torch.multiprocessing.get_context("fork")

        if args.task_params.dataset == "musdb":
            self.train_dataset = MusdbTrainDataset(
                target=args.task_params.target,
                root=args.dir_params.root,
                seq_duration=args.data_params.seq_dur,
                samples_per_track=args.data_params.samples_per_track,
                source_augmentations=aug_from_str(
                    ["gain", "channelswap"],
                ),
                sample_rate=args.data_params.sample_rate,
                seed=args.sys_params.seed,
                limitaug_method=args.data_params.limitaug_method,
                limitaug_mode=args.data_params.limitaug_mode,
                limitaug_custom_target_lufs=args.data_params.limitaug_custom_target_lufs,
                limitaug_custom_target_lufs_std=args.data_params.limitaug_custom_target_lufs_std,
                target_loudnorm_lufs=args.data_params.target_loudnorm_lufs,
                custom_limiter_attack_range=args.data_params.custom_limiter_attack_range,
                custom_limiter_release_range=args.data_params.custom_limiter_release_range,
            )
            self.valid_dataset = MusdbValidDataset(
                target=args.task_params.target, root=args.dir_params.root
            )
        elif args.task_params.dataset == "delimit":
            if args.data_params.limitaug_method == "ozone":
                self.train_dataset = OzoneTrainDataset(
                    target=args.task_params.target,
                    root=args.dir_params.root,
                    ozone_root=args.dir_params.ozone_root,
                    use_fixed=args.data_params.use_fixed,
                    seq_duration=args.data_params.seq_dur,
                    samples_per_track=args.data_params.samples_per_track,
                    source_augmentations=aug_from_str(
                        ["gain", "channelswap"],
                    ),
                    sample_rate=args.data_params.sample_rate,
                    seed=args.sys_params.seed,
                    limitaug_method=args.data_params.limitaug_method,
                    limitaug_mode=args.data_params.limitaug_mode,
                    limitaug_custom_target_lufs=args.data_params.limitaug_custom_target_lufs,
                    limitaug_custom_target_lufs_std=args.data_params.limitaug_custom_target_lufs_std,
                    target_loudnorm_lufs=args.data_params.target_loudnorm_lufs,
                    target_limitaug_mode=args.data_params.target_limitaug_mode,
                    target_limitaug_custom_target_lufs=args.data_params.target_limitaug_custom_target_lufs,
                    target_limitaug_custom_target_lufs_std=args.data_params.target_limitaug_custom_target_lufs_std,
                    custom_limiter_attack_range=args.data_params.custom_limiter_attack_range,
                    custom_limiter_release_range=args.data_params.custom_limiter_release_range,
                )
                self.valid_dataset = OzoneValidDataset(
                    target=args.task_params.target,
                    root=args.dir_params.root,
                    ozone_root=args.dir_params.ozone_root,
                    target_loudnorm_lufs=args.data_params.target_loudnorm_lufs,
                )
            else:
                self.train_dataset = DelimitTrainDataset(
                    target=args.task_params.target,
                    root=args.dir_params.root,
                    seq_duration=args.data_params.seq_dur,
                    samples_per_track=args.data_params.samples_per_track,
                    source_augmentations=aug_from_str(
                        ["gain", "channelswap"],
                    ),
                    sample_rate=args.data_params.sample_rate,
                    seed=args.sys_params.seed,
                    limitaug_method=args.data_params.limitaug_method,
                    limitaug_mode=args.data_params.limitaug_mode,
                    limitaug_custom_target_lufs=args.data_params.limitaug_custom_target_lufs,
                    limitaug_custom_target_lufs_std=args.data_params.limitaug_custom_target_lufs_std,
                    target_loudnorm_lufs=args.data_params.target_loudnorm_lufs,
                    target_limitaug_mode=args.data_params.target_limitaug_mode,
                    target_limitaug_custom_target_lufs=args.data_params.target_limitaug_custom_target_lufs,
                    target_limitaug_custom_target_lufs_std=args.data_params.target_limitaug_custom_target_lufs_std,
                    custom_limiter_attack_range=args.data_params.custom_limiter_attack_range,
                    custom_limiter_release_range=args.data_params.custom_limiter_release_range,
                )
                self.valid_dataset = DelimitValidDataset(
                    target=args.task_params.target,
                    root=args.dir_params.root,
                    delimit_valid_root=args.dir_params.delimit_valid_root,
                    valid_target_lufs=args.data_params.valid_target_lufs,
                    target_loudnorm_lufs=args.data_params.target_loudnorm_lufs,
                    delimit_valid_L_root=args.dir_params.delimit_valid_L_root,
                )

        self.train_sampler = DistributedSampler(
            self.train_dataset, shuffle=True, rank=args.gpu
        )
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=args.hyperparams.batch_size,
            shuffle=False,
            num_workers=args.sys_params.nb_workers,
            multiprocessing_context=self.mp_context,
            pin_memory=True,
            sampler=self.train_sampler,
            drop_last=False,
        )

        self.valid_sampler = DistributedSampler(
            self.valid_dataset, shuffle=False, rank=args.gpu
        )
        self.valid_loader = torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.sys_params.nb_workers,
            multiprocessing_context=self.mp_context,
            pin_memory=False,
            sampler=self.valid_sampler,
            drop_last=False,
        )

    def train(self, args, epoch):
        self.end = time.time()
        self.model.train()

        # get current learning rate
        for param_group in self.optimizer.param_groups:
            current_lr = param_group["lr"]

        if (
            args.sys_params.rank % args.ngpus_per_node == 0
        ):  # when the last rank process is finished
            print(f"Epoch {epoch}, Learning rate: {current_lr}")

        losses = utils.AverageMeter()
        loss_logger = {}

        loss_logger["train/train loss"] = 0
        # with torch.autograd.detect_anomaly():  # use this if you want to detect anomaly behavior while training.
        for i, values in enumerate(self.train_loader):
            mixture, clean, *train_vars = values

            mixture = mixture.cuda(args.gpu, non_blocking=True)
            clean = clean.cuda(args.gpu, non_blocking=True)
            target = clean  # target_shape = [batch_size, n_srcs, nb_channels (if stereo: 2), wave_length]
            loss_input = {}

            estimates, *estimates_vars = self.model(mixture)
            # estimates = self.model(mixture)

            # loss = []
            dict_loss = {}

            if args.task_params.dataset == "delimit":
                estimates = estimates_vars[0]

            for train_loss_idx, single_train_loss_func in enumerate(
                args.model_loss_params.train_loss_func
            ):
                if self.model.module.use_encoder_to_target:
                    target_spec = self.model.module.encoder(
                        rearrange(target, "b s c t -> (b s) c t")
                    )
                    target_spec = rearrange(
                        target_spec,
                        "(b s) c f t -> b s c f t",
                        s=args.task_params.bleeding_nsrcs,
                    )
                loss_else = self.criterion[single_train_loss_func](
                    estimates,
                    target_spec
                    if self.model.module.use_encoder_to_target
                    else target,
                )
                dict_loss[single_train_loss_func] = (
                    loss_else.mean()
                    * args.model_loss_params.train_loss_scales[train_loss_idx]
                )

            loss = sum([value for key, value in dict_loss.items()])

            ############################################################

            #################### 5. Back propagation ####################
            loss.backward()
            if args.hyperparams.gradient_clip:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=args.hyperparams.gradient_clip
                )

            losses.update(loss.item(), clean.size(0))

            loss_logger["train/train loss"] = losses.avg
            for key, value in dict_loss.items():
                loss_logger[f"train/{key}"] = value.item()

            self.optimizer.step()

            self.model.zero_grad(
                set_to_none=True
            )  # set_to_none=True is for memory saving

            if args.hyperparams.ema:
                self.model_ema.update()
            ############################################################

            # ###################### 6. Plot ######################

            if i % 30 == 0:
                # loss print for multiple loss function
                multiple_score = torch.Tensor(
                    [value for key, value in loss_logger.items()]
                ).to(args.gpu)
                gathered_score_list = [
                    torch.ones_like(multiple_score)
                    for _ in range(dist.get_world_size())
                ]
                dist.all_gather(gathered_score_list, multiple_score)
                gathered_score = torch.mean(
                    torch.stack(gathered_score_list, dim=0), dim=0
                )
                if args.gpu == 0:
                    print(f"Epoch {epoch},  step {i} / {len(self.train_loader)}")
                    temp_loss_logger = {}
                    for index, (key, value) in enumerate(loss_logger.items()):
                        temp_key = key.replace("train/", "iter-wise/")
                        temp_loss_logger[temp_key] = round(
                            gathered_score[index].item(), 6
                        )
                        print(f"{key} : {round(gathered_score[index].item(), 6)}")

        single_score = torch.Tensor([losses.avg]).to(args.gpu)

        gathered_score_list = [
            torch.ones_like(single_score) for _ in range(dist.get_world_size())
        ]
        dist.all_gather(gathered_score_list, single_score)
        gathered_score = torch.mean(torch.cat(gathered_score_list)).item()
        if args.gpu == 0:
            self.train_losses.append(gathered_score)
            if args.wandb_params.use_wandb:
                loss_logger["train/train loss"] = single_score
                loss_logger["train/epoch"] = epoch
                wandb.log(loss_logger)
            ############################################################

    def multi_validate(self, args, epoch):
        if args.gpu == 0:
            print(f"Epoch {epoch} Validation session!")

        losses = utils.AverageMeter()

        loss_logger = {}

        self.model.eval()

        with torch.no_grad():
            for i, values in enumerate(self.valid_loader, start=1):
                mixture, clean, song_name, *valid_vars = values

                mixture = mixture.cuda(args.gpu, non_blocking=True)
                clean = clean.cuda(args.gpu, non_blocking=True)
                target = clean

                dict_loss = {}
                if not args.data_params.singleset_num_frames:
                    if args.hyperparams.ema:
                        estimates, *estimates_vars = self.model_ema(mixture)
                    else:
                        estimates, *estimates_vars = self.model(mixture)
                    if args.task_params.dataset == "delimit":
                        estimates = estimates_vars[0]

                    estimates = estimates[..., : clean.size(-1)]

                else:  # use SingleTrackSet
                    db = SingleTrackSet(
                        mixture[0],
                        hop_length=args.data_params.nhop,
                        num_frame=args.data_params.singleset_num_frames,
                        target_name=args.task_params.target,
                    )
                    separated = []

                    for item in db:

                        if args.hyperparams.ema:
                            estimates, *estimates_vars = self.model_ema(
                                item.unsqueeze(0).to(args.gpu)
                            )
                        else:
                            estimates, *estimates_vars = self.model(
                                item.unsqueeze(0).to(args.gpu)
                            )

                        if args.task_params.dataset == "delimit":
                            estimates = estimates_vars[0]

                        separated.append(
                            estimates_vars[0][
                                ..., db.trim_length : -db.trim_length
                            ].clone()
                        )

                    estimates = torch.cat(separated, dim=-1)
                    estimates = estimates[..., : target.shape[-1]]

                for valid_loss_idx, single_valid_loss_func in enumerate(
                    args.model_loss_params.valid_loss_func
                ):
                    loss_else = self.criterion[single_valid_loss_func](
                        estimates,
                        target,
                    )
                    dict_loss[single_valid_loss_func] = (
                        loss_else.mean()
                        * args.model_loss_params.valid_loss_scales[valid_loss_idx]
                    )

                loss = sum([value for key, value in dict_loss.items()])

                losses.update(loss.item(), clean.size(0))

            list_sum_count = torch.Tensor([losses.sum, losses.count]).to(args.gpu)
            list_gathered_sum_count = [
                torch.ones_like(list_sum_count) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(list_gathered_sum_count, list_sum_count)
            gathered_score = reduce(
                torch.stack(list_gathered_sum_count), "s c -> c", "sum"
            )  # s: sum of losses.sum, c: sum of losses.count
            gathered_score = (gathered_score[0] / gathered_score[1]).item()

            loss_logger["valid/valid loss"] = gathered_score
            for key, value in dict_loss.items():
                loss_logger[f"valid/{key}"] = value.item()

            if args.hyperparams.lr_scheduler == "step_lr":
                self.scheduler.step(gathered_score)
            elif args.hyperparams.lr_scheduler == "cos_warmup":
                self.scheduler.step(epoch)
            else:
                self.scheduler.step(gathered_score)

            if args.wandb_params.use_wandb and args.gpu == 0:
                loss_logger["valid/epoch"] = epoch
                wandb.log(loss_logger)

            if args.gpu == 0:
                self.valid_losses.append(gathered_score)

                self.stop = self.es.step(gathered_score)

                print(f"Epoch {epoch}, validation loss : {round(gathered_score, 6)}")

                plt.plot(self.train_losses, label="train loss")
                plt.plot(self.valid_losses, label="valid loss")
                plt.legend(loc="upper right")
                plt.savefig(f"{args.output}/loss_graph_{args.task_params.target}.png")
                plt.close()

                save_states = {
                    "epoch": epoch,
                    "state_dict": self.model.module.state_dict()
                    if not args.hyperparams.ema
                    else self.model_ema.state_dict(),
                    "best_loss": self.es.best,
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                }

                utils.save_checkpoint(
                    save_states,
                    state_dict_only=gathered_score == self.es.best,
                    path=args.output,
                    target=args.task_params.target,
                )

                self.train_times.append(time.time() - self.end)

                if gathered_score == self.es.best:
                    self.best_epoch = epoch

                # save params
                params = {
                    "epochs_trained": epoch,
                    "args": args.toDict(),
                    "best_loss": self.es.best,
                    "best_epoch": self.best_epoch,
                    "train_loss_history": self.train_losses,
                    "valid_loss_history": self.valid_losses,
                    "train_time_history": self.train_times,
                    "num_bad_epochs": self.es.num_bad_epochs,
                }

                with open(
                    f"{args.output}/{args.task_params.target}.json", "w"
                ) as outfile:
                    outfile.write(json.dumps(params, indent=4, sort_keys=True))

                self.train_times.append(time.time() - self.end)
                print(
                    f"Epoch {epoch} train completed. Took {round(self.train_times[-1], 3)} seconds"
                )

    def resume(self, args):
        print(f"Resume checkpoint from: {args.dir_params.resume}:")
        loc = f"cuda:{args.gpu}"
        checkpoint_path = f"{args.dir_params.resume}/{args.task_params.target}"
        with open(f"{checkpoint_path}.json", "r") as stream:
            results = json.load(stream)
        checkpoint = torch.load(f"{checkpoint_path}.chkpnt", map_location=loc)

        if args.hyperparams.ema:
            self.model_ema.load_state_dict(checkpoint["state_dict"])
        else:
            self.model.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        if (
            args.dir_params.continual_train
        ):  # we want to use a pre-trained model but not want to use lr_scheduler history.
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = args.hyperparams.lr
        else:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            self.es.best = results["best_loss"]
            self.es.num_bad_epochs = results["num_bad_epochs"]

        self.start_epoch = results["epochs_trained"]
        self.train_losses = results["train_loss_history"]
        self.valid_losses = results["valid_loss_history"]
        self.train_times = results["train_time_history"]
        self.best_epoch = results["best_epoch"]
        if args.sys_params.rank % args.ngpus_per_node == 0:
            print(
                f"=> loaded checkpoint {checkpoint_path} (epoch {results['epochs_trained']})"
            )

    def cal_loss(self, args, loss_input):
        loss_dict = {}
        for key, value in loss_input.items():
            loss_dict[key] = self.criterion[key](*value)

        return loss_dict

    def cal_multiple_losses(self, args, dict_loss_name_input):
        loss_dict = {}
        for loss_name, loss_input in dict_loss_name_input.items():
            loss_dict[loss_name] = self.cal_loss(args, loss_input)

        return loss_dict
