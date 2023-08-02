import os
import glob
import argparse
import csv

import numpy as np
import librosa
import soundfile as sf
import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="Save sample-wise gain parameters for dataset distribution"
    )
    parser.add_argument(
        "--root",
        type=str,
        default="/path/to/musdb18hq",
        help="Root directory",
    )
    parser.add_argument(
        "--musdb_XL_train_root",
        type=str,
        default="/path/to/musdb-XL-train",
        help="Directory of musdb-XL-train dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/path/to/musdb-XL-train/np_ratio",
        help="Directory to save sample-wise gain ratio",
    )

    args = parser.parse_args()

    sources = ["vocals", "bass", "drums", "other"]

    path_csv_fixed = f"{args.musdb_XL_train_root}/ozone_train_fixed.csv"
    list_path_csv_random = sorted(
        glob.glob(f"{args.musdb_XL_train_root}/ozone_train_random_*.csv")
    )

    # read ozone_train_fixed list
    fixed_list = []
    os.makedirs(f"{args.output}/ozone_train_fixed", exist_ok=True)
    with open(path_csv_fixed, "r", encoding="utf-8") as f:
        rdr = csv.reader(f)
        for k, line in enumerate(rdr):
            if k == 0:  # song_name, max_threshold, max_character
                pass
            else:
                fixed_list.append(line)

    # save numpy files of ozone_train_fixed
    # which is the limiter-applied version of 100 songs from musdb-HQ train set
    # each numpy file contain sample-wise gain ratio parameters
    for fixed_song in tqdm.tqdm(fixed_list):
        audio_sources = []
        for source in sources:
            audio, sr = librosa.load(
                f"{args.root}/train/{fixed_song[0]}/{source}.wav", sr=44100, mono=False
            )
            audio_sources.append(audio)
        stems = np.stack(audio_sources, axis=0)
        mixture = stems.sum(0)

        ozone_mixture, sr = librosa.load(
            f"{args.musdb_XL_train_root}/ozone_train_fixed/{fixed_song[0]}.wav",
            sr=44100,
            mono=False,
        )
        mixture[mixture == 0.0] = np.finfo(np.float32).eps  # to avoid 'divided by zero'
        ratio = ozone_mixture / mixture

        np.save(
            f"{args.output}/ozone_train_fixed/{fixed_song[0]}.npy",
            ratio.astype(np.float16),  # 16bit is enough...
        )

    # read ozone_train_random list
    random_list = []
    os.makedirs(f"{args.output}/ozone_train_random", exist_ok=True)
    for path_csv_random in list_path_csv_random:
        with open(path_csv_random, "r", encoding="utf-8") as f:
            rdr = csv.reader(f)
            for k, line in enumerate(rdr):
                if k == 0:
                    # ['song_name',
                    #  'max_threshold',
                    #  'max_character',
                    #  'vocals_name',
                    #  'vocals_start_sec',
                    #  'vocals_gain',
                    #  'vocals_channelswap',
                    #  'bass_name',
                    #  'bass_start_sec',
                    #  'bass_gain',
                    #  'bass_channelswap',
                    #  'drums_name',
                    #  'drums_start_sec',
                    #  'drums_gain',
                    #  'drums_channelswap',
                    #  'other_name',
                    #  'other_start_sec',
                    #  'other_gain',
                    #  'other_channelswap']
                    pass
                else:
                    random_list.append(line)

    # save wave files of ozone_train_random,
    # which is the limiter-applied version of 4-sec 300,000 segments randomly created from musdb-HQ train subset
    for random_song in tqdm.tqdm(random_list):
        audio_sources = []
        for k, source in enumerate(sources):
            audio, sr = librosa.load(
                f"{args.root}/train/{random_song[3 + k * 4]}/{source}.wav",
                sr=44100,
                mono=False,
                offset=float(random_song[4 + k * 4]),  # 'inst_start_sec'
                duration=4.0,
            )
            audio = audio * float(random_song[5 + k * 4])  # 'inst_gain'
            if random_song[6 + k * 4].lower() == "true":  # 'inst_channelswap'
                audio = np.flip(audio, axis=0)

            audio_sources.append(audio)
        stems = np.stack(audio_sources, axis=0)
        mixture = stems.sum(0)

        ozone_mixture, sr = librosa.load(
            f"{args.musdb_XL_train_root}/ozone_train_random/{random_song[0]}.wav",
            sr=44100,
            mono=False,
        )

        mixture[mixture == 0.0] = np.finfo(np.float32).eps  # to avoid 'divided by zero'
        ratio = ozone_mixture / mixture

        np.save(
            f"{args.output}/ozone_train_random/{random_song[0]}.npy",
            ratio.astype(np.float16),  # 16bit is enough...
        )


if __name__ == "__main__":
    main()
