import os
import json
import csv
import glob
import argparse
import random
import math

import librosa
import soundfile as sf
import pedalboard
import numpy as np
import pyloudnorm as pyln
from scipy.stats import gamma
import torchaudio


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def _augment_gain_ozone(audio, low=0.25, high=1.25):
    """Applies a random gain between `low` and `high`"""
    g = low + random.random() * (high - low)
    return audio * g, g


def _augment_channelswap_ozone(audio):
    """Swap channels of stereo signals with a probability of p=0.5"""
    if audio.shape[0] == 2 and random.random() < 0.5:
        return np.flip(audio, axis=0), True  # axis=0 must be given
    else:
        return audio, False


# load wav file from arbitrary positions of 16bit stereo wav file
def load_wav_arbitrary_position_stereo(
    filename, sample_rate, seq_duration, return_pos=False
):
    # stereo
    # seq_duration[second]
    length = torchaudio.info(filename).num_frames

    random_start = random.randint(
        0, int(length - math.ceil(seq_duration * sample_rate) - 1)
    )
    random_start_sec = librosa.samples_to_time(random_start, sr=sample_rate)
    X, sr = librosa.load(
        filename, sr=None, mono=False, offset=random_start_sec, duration=seq_duration
    )

    if return_pos:
        return X, random_start_sec
    else:
        return X


# def main():
parser = argparse.ArgumentParser(description="Preprocess audio files for training")
parser.add_argument(
    "--root",
    type=str,
    default="/Users/jeon/Desktop/pythonpractice/musdb18hq",
    help="Root directory",
)
parser.add_argument(
    "--output",
    type=str,
    # default="/Users/jeon/Desktop/pythonpractice/delimit/data",
    default="/Volumes/Samsung_T5/personal/data/delimit",
    help="Where to save output files",
)
parser.add_argument(
    "--n_samples", type=int, default=300000, help="Number of samples to save"
)
parser.add_argument("--seq_duration", type=float, default=4.0, help="Sequence duration")
parser.add_argument(
    "--save_fixed", type=str2bool, default=False, help="Save fixed mixture audio"
)
parser.add_argument(
    "--target_lufs_mean", type=float, default=-8.0, help="Target LUFS mean"
)
parser.add_argument(
    "--target_lufs_std", type=float, default=-1.0, help="Target LUFS std"
)
parser.add_argument("--sample_rate", type=int, default=44100, help="Sample rate")
parser.add_argument("--seed", type=int, default=46, help="Random seed")
args = parser.parse_args()
random.seed(args.seed)

valid_list = [
    "ANiMAL - Rockshow",
    "Actions - One Minute Smile",
    "Alexander Ross - Goodbye Bolero",
    "Clara Berry And Wooldog - Waltz For My Victims",
    "Fergessen - Nos Palpitants",
    "James May - On The Line",
    "Johnny Lokke - Promises & Lies",
    "Leaf - Summerghost",
    "Meaxic - Take A Step",
    "Patrick Talbot - A Reason To Leave",
    "Skelpolu - Human Mistakes",
    "Traffic Experiment - Sirens",
    "Triviul - Angelsaint",
    "Young Griffo - Pennies",
]

meter = pyln.Meter(args.sample_rate)


sources = ["vocals", "bass", "drums", "other"]
song_list = glob.glob(f"{args.root}/train/*")

vst = pedalboard.load_plugin(
    "/Library/Audio/Plug-Ins/Components/iZOzone9ElementsAUHook.component"
)

if args.save_fixed:
    vst_params = []

    os.makedirs(f"{args.output}/ozone_train_fixed", exist_ok=True)

    for song in song_list:
        print(f"Processing {song}...")
        song_name = os.path.basename(song)
        audio_sources = []
        for source in sources:
            audio_path = f"{song}/{source}.wav"
            audio, sr = librosa.load(audio_path, sr=args.sample_rate, mono=False)
            audio_sources.append(audio)
        stems = np.stack(audio_sources, axis=0)
        mixture = stems.sum(0)
        lufs = meter.integrated_loudness(mixture.T)
        target_lufs = random.gauss(args.target_lufs_mean, args.target_lufs_std)
        adjusted_loudness = target_lufs - lufs

        vst.reset()
        vst.eq_bypass = True
        vst.img_bypass = True
        vst.max_mode = 1.0  # Set IRC2 mode
        vst.max_threshold = min(-adjusted_loudness, 0.0)
        vst.max_character = min(gamma.rvs(2), 10.0)

        print(
            f"Applying Ozone 9 Elements IRC2 with threshold {vst.max_threshold} and character {vst.max_character}..."
        )
        limited_mixture = vst(mixture, args.sample_rate)

        sf.write(
            f"{args.output}/ozone_train_fixed/{song_name}.wav",
            limited_mixture.T,
            args.sample_rate,
        )
        vst_params.append([song_name, vst.max_threshold, vst.max_character])
        # Save the song name and vst parameters (vst.max_threshold and vst.max_character) to a csv file
        with open(f"{args.output}/ozone_train_fixed.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(["song_name", "max_threshold", "max_character"])
            for idx, list_vst_param in enumerate(vst_params):
                writer.writerow(list_vst_param)

else:
    if os.path.exists(f"{args.output}/ozone_train_random_0.csv"):
        vst_params = []
        list_csv_files = glob.glob(f"{args.output}/ozone_train_random_*.csv")
        list_csv_files.sort()
        for csv_file in list_csv_files:
            with open(csv_file, "r") as f:
                reader = csv.reader(f)
                next(reader)
                vst_params.extend([row for row in reader])

        # Load the song name and vst parameters (vst.max_threshold and vst.max_character) from a csv file
        # vst_dict = {}
        # with open(f"{args.output}/ozone_train_random.csv", "r") as f:
        #     reader = csv.reader(f)
        #     next(reader)
        #     vst_params = [row for row in reader]
        # for row in reader:
        #     vst_dict[row[0]] = {
        #         "max_threshold": float(row[1]),
        #         "max_character": float(row[2]),
        #         "vocals": {
        #             "name": row[3],
        #             "start_sec": float(row[4]),
        #             "gain": float(row[5]),
        #             "channelswap": bool(row[6]),
        #         },
        #         "bass": {
        #             "name": row[7],
        #             "start_sec": float(row[8]),
        #             "gain": float(row[9]),
        #             "channelswap": bool(row[10]),
        #         },
        #         "drums": {
        #             "name": row[11],
        #             "start_sec": float(row[12]),
        #             "gain": float(row[13]),
        #             "channelswap": bool(row[14]),
        #         },
        #         "other": {
        #             "name": row[15],
        #             "start_sec": float(row[16]),
        #             "gain": float(row[17]),
        #             "channelswap": bool(row[18]),
        #         },
        #     }
    # elif os.path.exists(f"{args.output}/ozone_train_random.json"):
    #     with open(f"{args.output}/ozone_train_random.json", "r") as f:
    #         vst_dict = json.load(f)
    else:
        vst_params = []
        # vst_dict = {}
    song_list = [x for x in song_list if os.path.basename(x) not in valid_list]

    os.makedirs(f"{args.output}/ozone_train_random", exist_ok=True)

    for n in range(len(vst_params), args.n_samples):
        print(f"Processing {n} / {args.n_samples}...")
        seg_name = f"ozone_seg_{n}"

        lufs_not_inf = True
        while lufs_not_inf:
            audio_sources = []
            source_song_names = {}
            source_start_secs = {}
            source_gains = {}
            source_channelswaps = {}
            for source in sources:
                track_path = random.choice(song_list)
                song_name = os.path.basename(track_path)
                audio_path = f"{track_path}/{source}.wav"
                audio, start_sec = load_wav_arbitrary_position_stereo(
                    audio_path, args.sample_rate, args.seq_duration, return_pos=True
                )
                audio, gain = _augment_gain_ozone(audio)
                audio, channelswap = _augment_channelswap_ozone(audio)
                audio_sources.append(audio)
                source_song_names[source] = song_name
                source_start_secs[source] = start_sec
                source_gains[source] = gain
                source_channelswaps[source] = channelswap

            stems = np.stack(audio_sources, axis=0)
            mixture = stems.sum(0)
            lufs = meter.integrated_loudness(mixture.T)

            # if lufs is inf, then the mixture is silent, so we need to generate a new mixture
            lufs_not_inf = np.isinf(lufs)

        target_lufs = random.gauss(args.target_lufs_mean, args.target_lufs_std)
        adjusted_loudness = target_lufs - lufs

        vst.reset()
        vst.eq_bypass = True
        vst.img_bypass = True
        vst.max_mode = 1.0  # Set IRC2 mode
        vst.max_threshold = min(max(-20, -adjusted_loudness), 0.0)
        vst.max_character = min(gamma.rvs(2), 10.0)

        print(
            f"Applying Ozone 9 Elements IRC2 with threshold {vst.max_threshold} and character {vst.max_character}..."
        )
        limited_mixture = vst(mixture, args.sample_rate)

        sf.write(
            f"{args.output}/ozone_train_random_0/{seg_name}.wav",
            limited_mixture.T,
            args.sample_rate,
        )
        vst_params.append(
            [
                seg_name,
                vst.max_threshold,
                vst.max_character,
                source_song_names["vocals"],
                source_start_secs["vocals"],
                source_gains["vocals"],
                source_channelswaps["vocals"],
                source_song_names["bass"],
                source_start_secs["bass"],
                source_gains["bass"],
                source_channelswaps["bass"],
                source_song_names["drums"],
                source_start_secs["drums"],
                source_gains["drums"],
                source_channelswaps["drums"],
                source_song_names["other"],
                source_start_secs["other"],
                source_gains["other"],
                source_channelswaps["other"],
            ]
        )
        # vst_dict[seg_name] = {
        #     "max_threshold": vst.max_threshold,
        #     "max_character": vst.max_character,
        #     "vocals": {
        #         "name": source_song_names["vocals"],
        #         "start_sec": source_start_secs["vocals"],
        #         "gain": source_gains["vocals"],
        #         "channelswap": source_channelswaps["vocals"],
        #     },
        #     "bass": {
        #         "name": source_song_names["bass"],
        #         "start_sec": source_start_secs["bass"],
        #         "gain": source_gains["bass"],
        #         "channelswap": source_channelswaps["bass"],
        #     },
        #     "drums": {
        #         "name": source_song_names["drums"],
        #         "start_sec": source_start_secs["drums"],
        #         "gain": source_gains["drums"],
        #         "channelswap": source_channelswaps["drums"],
        #     },
        #     "other": {
        #         "name": source_song_names["other"],
        #         "start_sec": source_start_secs["other"],
        #         "gain": source_gains["other"],
        #         "channelswap": source_channelswaps["other"],
        #     },
        # }

        if (n + 1) % 20000 == 0 or n == args.n_samples - 1:
            # We will separate the csv file into multiple files to avoid memory error
            # Save the song name and vst parameters (vst.max_threshold and vst.max_character) to a csv file
            number = int(n // 20000)
            with open(f"{args.output}/ozone_train_random_{number}.csv", "w") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "song_name",
                        "max_threshold",
                        "max_character",
                        "vocals_name",
                        "vocals_start_sec",
                        "vocals_gain",
                        "vocals_channelswap",
                        "bass_name",
                        "bass_start_sec",
                        "bass_gain",
                        "bass_channelswap",
                        "drums_name",
                        "drums_start_sec",
                        "drums_gain",
                        "drums_channelswap",
                        "other_name",
                        "other_start_sec",
                        "other_gain",
                        "other_channelswap",
                    ]
                )
                for idx, list_vst_param in enumerate(
                    vst_params[number * 20000 : (number + 1) * 20000]
                ):
                    writer.writerow(list_vst_param)
            # with open(f"{args.output}/ozone_train_random.json", "w") as outfile:
            #     outfile.write(json.dumps(vst_dict, indent=4, sort_keys=True))


# if __name__ == "__main__":
#     main()
