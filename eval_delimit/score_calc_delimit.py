# Calculate SI-SDR, Multi-resolution spectrogram mse score of the pre-inferenced sources
import os
import argparse
import csv
import json
import glob

import tqdm
import numpy as np
import librosa
import pyloudnorm as pyln
from asteroid.metrics import get_metrics

from utils import str2bool


def multi_resolution_spectrogram_mse(
    gt, est, n_fft=[2048, 1024, 512], n_hop=[512, 256, 128]
):
    assert gt.shape == est.shape
    assert len(n_fft) == len(n_hop)

    score = 0.0
    for i in range(len(n_fft)):
        gt_spec = librosa.magphase(
            librosa.stft(gt, n_fft=n_fft[i], hop_length=n_hop[i])
        )[0]
        est_spec = librosa.magphase(
            librosa.stft(est, n_fft=n_fft[i], hop_length=n_hop[i])
        )[0]
        score = score + np.mean((gt_spec - est_spec) ** 2)

    return score


parser = argparse.ArgumentParser(description="model test.py")

parser.add_argument(
    "--target",
    type=str,
    default="all",
    help="target source. all, vocals, drums, bass, other, 0.5_mixed",
)
parser.add_argument(
    "--root", type=str, default="/path/to/musdb18hq_loudnorm"
)
parser.add_argument("--exp_name", type=str, default="convtasnet_6_s")
parser.add_argument(
    "--output_directory",
    type=str,
    default="/path/to/results",
)
parser.add_argument("--loudnorm_lufs", type=float, default=-14.0)
parser.add_argument(
    "--calc_mse",
    type=str2bool,
    default=True,
    help="calculate multi-resolution spectrogram mse",
)

parser.add_argument(
    "--calc_results",
    type=str2bool,
    default=True,
    help="Set this True when you want to calculate the results of the test set. Set this False when calculating musdb-hq vs musdb-XL. (top row in Table 1.)",
)

args, _ = parser.parse_known_args()

args.sample_rate = 44100

meter = pyln.Meter(args.sample_rate)

if args.calc_results:
    args.test_output_dir = f"{args.output_directory}/test/{args.exp_name}"
else:
    args.test_output_dir = f"{args.output_directory}/{args.exp_name}"

if args.target == "all" or args.target == "0.5_mixed":
    test_tracks = glob.glob(f"{args.root}/*/mixture.wav")
else:
    test_tracks = glob.glob(f"{args.root}/*/{args.target}.wav")
i = 0

dict_song_score = {}
list_si_sdr = []
list_multi_mse = []
for track in tqdm.tqdm(test_tracks):
    if args.target == "all":  # for standard de-limiter estimation
        audio_name = os.path.basename(os.path.dirname(track))
        gt_source = librosa.load(track, sr=args.sample_rate, mono=False)[0]

        est_delimiter = librosa.load(
            f"{args.test_output_dir}/{audio_name}/all.wav",
            sr=args.sample_rate,
            mono=False,
        )[0]

    else:  # for source-separated de-limiter estimation
        audio_name = os.path.basename(os.path.dirname(track))
        gt_source = librosa.load(track, sr=args.sample_rate, mono=False)[0]
        est_delimiter = librosa.load(
            f"{args.test_output_dir}/{audio_name}/{args.target}.wav",
            sr=args.sample_rate,
            mono=False,
        )[0]


    metrics_dict = get_metrics(
        gt_source + est_delimiter,
        gt_source,
        est_delimiter,
        sample_rate=args.sample_rate,
        metrics_list=["si_sdr"],
    )

    if args.calc_mse:
        multi_resolution_spectrogram_mse_score = multi_resolution_spectrogram_mse(
            gt_source, est_delimiter
        )
    else:
        multi_resolution_spectrogram_mse_score = None

    dict_song_score[audio_name] = {
        "si_sdr": metrics_dict["si_sdr"],
        "multi_mse": multi_resolution_spectrogram_mse_score,
    }
    list_si_sdr.append(metrics_dict["si_sdr"])
    list_multi_mse.append(multi_resolution_spectrogram_mse_score)

    i += 1

print(f"{args.exp_name} on {args.target}")
print(f"SI-SDR score: {sum(list_si_sdr) / len(list_si_sdr)}")
if args.calc_mse:
    print(f"multi-mse score: {sum(list_multi_mse) / len(list_multi_mse)}")

if args.target != "all":
    # save dict_song_score to json file
    with open(f"{args.test_output_dir}/score_{args.target}.json", "w") as f:
        json.dump(dict_song_score, f, indent=4)
else:
    # save dict_song_score to json file
    with open(f"{args.test_output_dir}/score.json", "w") as f:
        json.dump(dict_song_score, f, indent=4)
