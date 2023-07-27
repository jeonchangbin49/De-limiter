# To be honest... this is not ddp.
import os
import json
import argparse
import glob

import torch
import tqdm
import musdb
import librosa
import soundfile as sf
import pyloudnorm as pyln
from dotmap import DotMap

from models import load_model_with_args
from separate_func import (
    conv_tasnet_separate,
)
from utils import str2bool, db2linear


tqdm.monitor_interval = 0


def separate_track_with_model(
    args, model, device, track_audio, track_name, meter, augmented_gain
):
    with torch.no_grad():
        if (
            args.model_loss_params.architecture == "conv_tasnet_mask_on_output"
            or args.model_loss_params.architecture == "conv_tasnet"
        ):
            estimates = conv_tasnet_separate(
                args,
                model,
                device,
                track_audio,
                track_name,
                meter=meter,
                augmented_gain=augmented_gain,
            )

        return estimates


def main():
    parser = argparse.ArgumentParser(description="model test.py")

    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--data_root", type=str, default="/data1/Music/musdb_XL")
    parser.add_argument(
        "--use_musdb",
        type=str2bool,
        default=True,
        help="Use musdb test data or just want to inference other samples?",
    )
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--manual_output_name", type=str, default=None)
    parser.add_argument(
        "--output_directory", type=str, default="/data2/personal/jeon/delimit/results"
    )
    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_arugment("--save_name_as_target", type=str2bool, default=True)
    parser.add_argument(
        "--loudnorm_input_lufs",
        type=float,
        default=None,
        help="If you want to use loudnorm, input target lufs",
    )
    parser.add_argument(
        "--use_singletrackset",
        type=str2bool,
        default=False,
        help="Use SingleTrackSet for X-UMX",
    )
    parser.add_argument(
        "--best_model",
        type=str2bool,
        default=True,
        help="Use best model or lastly saved model",
    )
    parser.add_argument(
        "--save_output_loudnorm",
        type=float,
        default=None,
        help="Save loudness normalized outputs or not. If you want to save, input target loudness",
    )
    parser.add_argument(
        "--save_mixed_output",
        type=float,
        default=None,
        help="Save original+delimited-estimation mixed output with a ratio of default 0.5 (orginal) and 1 - 0.5 (estimation)",
    )
    parser.add_argument(
        "--save_16k_mono",
        type=str2bool,
        default=False,
        help="Save 16k mono wav files for FAD evaluation.",
    )
    parser.add_argument(
        "--save_histogram",
        type=str2bool,
        default=False,
        help="Save histogram of the output. Only valid when the task is 'delimit'",
    )

    args, _ = parser.parse_known_args()

    args.output_dir = f"{args.output_directory}/checkpoint/{args.exp_name}"
    with open(f"{args.output_dir}/{args.target}.json", "r") as f:
        args_dict = json.load(f)
        args_dict = DotMap(args_dict)

    for key, value in args_dict["args"].items():
        if key in list(vars(args).keys()):
            pass
        else:
            setattr(args, key, value)

    args.test_output_dir = f"{args.output_directory}/test/{args.exp_name}"

    if args.manual_output_name != None:
        args.test_output_dir = f"{args.output_directory}/test/{args.manual_output_name}"
    os.makedirs(args.test_output_dir, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.use_gpu else "cpu"
    )

    ###################### Define Models ######################
    our_model = load_model_with_args(args)
    our_model = our_model.to(device)
    print(our_model)
    pytorch_total_params = sum(
        p.numel() for p in our_model.parameters() if p.requires_grad
    )
    print("Total number of parameters", pytorch_total_params)
    # Future work => Torchinfo would be better for this purpose.

    if args.best_model:
        target_model_path = f"{args.output_dir}/{args.target}.pth"
        checkpoint = torch.load(target_model_path, map_location=device)
        our_model.load_state_dict(checkpoint)
    else:  # when using lastly saved model
        target_model_path = f"{args.output_dir}/{args.target}.chkpnt"
        checkpoint = torch.load(target_model_path, map_location=device)
        our_model.load_state_dict(checkpoint["state_dict"])

    our_model.eval()

    meter = pyln.Meter(44100)

    if args.use_musdb:
        test_tracks = musdb.DB(root=args.data_root, subsets="test", is_wav=True)

        for track in tqdm.tqdm(test_tracks):
            track_name = track.name
            track_audio = track.audio

            orig_audio = track_audio.copy()

            augmented_gain = None
            print("Now De-limiting : ", track_name)

            if args.loudnorm_input_lufs:  # If you want to use loud-normalized input
                track_lufs = meter.integrated_loudness(track_audio)
                augmented_gain = args.loudnorm_input_lufs - track_lufs
                track_audio = track_audio * db2linear(augmented_gain, eps=0.0)

            track_audio = (
                torch.as_tensor(track_audio.T, dtype=torch.float32)
                .unsqueeze(0)
                .to(device)
            )

            estimates = separate_track_with_model(
                args, our_model, device, track_audio, track_name, meter, augmented_gain
            )

            if args.save_mixed_output:
                orig_audio = orig_audio.T
                track_lufs = meter.integrated_loudness(orig_audio.T)
                augmented_gain = args.save_output_loudnorm - track_lufs
                orig_audio = orig_audio * db2linear(augmented_gain, eps=0.0)

                mixed_output = orig_audio * args.save_mixed_output + estimates * (
                    1 - args.save_mixed_output
                )

                sf.write(
                    f"{args.test_output_dir}/{track_name}/{str(args.save_mixed_output)}_mixed.wav",
                    mixed_output.T,
                    args.data_params.sample_rate,
                )
    else:
        test_tracks = glob.glob(f"{args.data_root}/*.wav") + glob.glob(
            f"{args.data_root}/*.mp3"
        )

        for track in tqdm.tqdm(test_tracks):
            track_name = os.path.basename(track).replace(".wav", "").replace(".mp3", "")
            track_audio, sr = librosa.load(
                track, sr=None, mono=False
            )  # sr should be 44100

            orig_audio = track_audio.copy()

            if sr != 44100:
                raise ValueError("Sample rate should be 44100")
            augmented_gain = None
            print("Now De-limiting : ", track_name)

            if args.loudnorm_input_lufs:  # If you want to use loud-normalized input
                track_lufs = meter.integrated_loudness(track_audio.T)
                augmented_gain = args.loudnorm_input_lufs - track_lufs
                track_audio = track_audio * db2linear(augmented_gain, eps=0.0)

            track_audio = (
                torch.as_tensor(track_audio, dtype=torch.float32)
                .unsqueeze(0)
                .to(device)
            )

            estimates = separate_track_with_model(
                args, our_model, device, track_audio, track_name, meter, augmented_gain
            )

            if args.save_mixed_output:
                track_lufs = meter.integrated_loudness(orig_audio.T)
                augmented_gain = args.save_output_loudnorm - track_lufs
                orig_audio = orig_audio * db2linear(augmented_gain, eps=0.0)

                mixed_output = orig_audio * args.save_mixed_output + estimates * (
                    1 - args.save_mixed_output
                )

                sf.write(
                    f"{args.test_output_dir}/{track_name}/{track_name}_mixed.wav",
                    mixed_output.T,
                    args.data_params.sample_rate,
                )


if __name__ == "__main__":
    main()
