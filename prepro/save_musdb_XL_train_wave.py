# Save musdb-XL-train dataset from numpy
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
        description="Save musdb-XL-train wave files from the downloaded sample-wise gain parameters"
    )
    parser.add_argument(
        "--root",
        type=str,
        default="/path/to/musdb18hq",
        help="Root directory",
    )
    parser.add_argument(
        "--musdb_XL_train_npy_root",
        type=str,
        default="/path/to/musdb-XL-train",
        help="Directory of numpy arrays of musdb-XL-train's sample-wise ratio ",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/path/to/musdb-XL-train",
        help="Directory to save musdb-XL-train wave data",
    )

    args = parser.parse_args()

    sources = ["vocals", "bass", "drums", "other"]

    path_csv_fixed = f"{args.musdb_XL_train_npy_root}/ozone_train_fixed.csv"
    list_path_csv_random = sorted(
        glob.glob(f"{args.musdb_XL_train_npy_root}/ozone_train_random_*.csv")
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

    # save wave files of ozone_train_fixed,
    # which is the limiter-applied version of 100 songs from musdb-HQ train set
    for fixed_song in tqdm.tqdm(fixed_list):
        audio_sources = []
        for source in sources:
            audio, sr = librosa.load(
                f"{args.root}/train/{fixed_song[0]}/{source}.wav", sr=44100, mono=False
            )
            audio_sources.append(audio)
        stems = np.stack(audio_sources, axis=0)
        mixture = stems.sum(0)

        ratio = np.load(
            f"{args.musdb_XL_train_npy_root}/np_ratio/ozone_train_fixed/{fixed_song[0]}.npy"
        )
        output = mixture * ratio

        sf.write(
            f"{args.output}/ozone_train_fixed/{fixed_song[0]}.wav",
            output.T,
            44100,
            subtype="PCM_16",
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

        ratio = np.load(
            f"{args.musdb_XL_train_npy_root}/np_ratio/ozone_train_random/{random_song[0]}.npy"
        )
        output = mixture * ratio

        sf.write(
            f"{args.output}/ozone_train_random/{random_song[0]}.wav",
            output.T,
            44100,
            subtype="PCM_16",
        )


if __name__ == "__main__":
    main()
