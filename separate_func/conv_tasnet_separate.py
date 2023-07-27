import os

import soundfile as sf
import torch
import pyloudnorm as pyln
import librosa
import matplotlib
import matplotlib.pyplot as plt

from dataloader import SingleTrackSet
from utils import db2linear


def conv_tasnet_separate(
    args, our_model, device, track_audio, track_name, meter=None, augmented_gain=None
):

    if args.use_singletrackset:
        db = SingleTrackSet(
            track_audio.squeeze(dim=0),
            hop_length=args.data_params.nhop,
            num_frame=128,
            target_name=args.target,
        )
        separated = []

        for item in db:
            item = item.unsqueeze(0).to(device)
            estimates, *estimates_vars = our_model(item)
            if args.task_params.dataset == "delimit":
                estimates = estimates_vars[0]

            estimates = estimates.cpu().detach()
            separated.append(
                estimates[..., db.trim_length : -db.trim_length].cpu().detach().clone()
            )

        estimates = torch.cat(separated, dim=-1)
        estimates = estimates[0, :, : track_audio.shape[-1]].numpy()
    else:
        estimates, *estimates_vars = our_model(track_audio)
        if args.save_histogram and args.task_params.dataset == "delimit":
            plt.figure(figsize=(10, 10))
            plt.hist(estimates.cpu().detach().numpy().flatten(), bins=100)
            os.makedirs(f"{args.test_output_dir}/{track_name}", exist_ok=True)
            plt.savefig(
                f"{args.test_output_dir}/{track_name}/{args.target}_histogram.png"
            )
        if args.task_params.dataset == "delimit":
            estimates = estimates_vars[0]

        estimates = estimates.cpu().detach().numpy()
        estimates = estimates[0, :, : track_audio.shape[-1]]

    if args.save_name_as_target:
        os.makedirs(f"{args.test_output_dir}/{track_name}", exist_ok=True)

    if args.save_output_loudnorm:
        print("SAVE Loudness normalized OUTPUT ")
        loudness = meter.integrated_loudness(estimates.T)
        estimates = estimates * db2linear(args.save_output_loudnorm - loudness, eps=0.0)
    elif augmented_gain != None and args.save_output_loudnorm == None:
        estimates = estimates * db2linear(-augmented_gain, eps=0.0)

    sf.write(
        f"{args.test_output_dir}/{track_name}/{args.target}.wav"
        if args.save_name_as_target
        else f"{args.test_output_dir}/{track_name}.wav",
        estimates.T,
        samplerate=args.data_params.sample_rate,
    )

    if args.save_16k_mono:
        estimates_16k_mono = librosa.to_mono(estimates)
        estimates_16k_mono = librosa.resample(
            estimates_16k_mono,
            orig_sr=args.data_params.sample_rate,
            target_sr=16000,
        )
        os.makedirs(f"{args.test_output_dir}_16k_mono/{track_name}", exist_ok=True)
        sf.write(
            f"{args.test_output_dir}_16k_mono/{track_name}/{args.target}.wav"
            if args.save_name_as_target
            else f"{args.test_output_dir}_16k_mono/{track_name}.wav",
            estimates_16k_mono,
            samplerate=16000,
        )

    return estimates
