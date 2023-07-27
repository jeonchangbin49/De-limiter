# Save loudness normalized (-14 LUFS) musdb-XL audio files for delimiter evaluation

import os
import argparse

import tqdm
import musdb
import soundfile as sf
import librosa
import pyloudnorm as pyln

from utils import db2linear, str2bool


tqdm.monitor_interval = 0


def main():
    parser = argparse.ArgumentParser(description="model test.py")

    parser.add_argument(
        "--target",
        type=str,
        default="mixture",
        help="target source. all, vocals, drums, bass, other",
    )
    parser.add_argument("--data_root", type=str, default="/data1/Music/musdb_XL")
    # parser.add_argument("--data_root", type=str, default="/data1/Music/musdb18hq")
    parser.add_argument(
        "--data_root_hq",
        type=str,
        default="/data1/Music/musdb18hq",
        help="this is used when saving loud-norm stem of musdb-XL")
    parser.add_argument(
        "--output_directory",
        type=str,
        default="/data2/personal/jeon/delimit/data/musdb_XL_loudnorm",
        # default="/data2/personal/jeon/delimit/data/musdb_hq_loudnorm",
    )
    parser.add_argument(
        "--loudnorm_input_lufs",
        type=float,
        default=-14.0,
        help="If you want to use loudnorm, input target lufs",
    )
    parser.add_argument(
        "--save_16k_mono",
        type=str2bool,
        default=True,
        help="Save 16k mono wav files for FAD evaluation.",
    )


    args, _ = parser.parse_known_args()

    os.makedirs(args.output_directory, exist_ok=True)

    meter = pyln.Meter(44100)

    test_tracks = musdb.DB(root=args.data_root, subsets="test", is_wav=True)
    if args.target != "mixture":
        hq_tracks = musdb.DB(root=args.data_root_hq, subsets='test', is_wav=True)

    for idx, track in tqdm.tqdm(enumerate(test_tracks)):
        track_name = track.name
        if (
            os.path.basename(args.data_root) == "musdb18hq"
            and track_name == "PR - Oh No"
        ):  # We have to consider this exception because 'PR - Oh No' mixture.wav is left-panned. We will use the linear mixture instead.
            # Please refer https://github.com/jeonchangbin49/musdb-XL/blob/main/make_L_and_XL.py
            track_audio = (
                track.targets["vocals"].audio
                + track.targets["drums"].audio
                + track.targets["bass"].audio
                + track.targets["other"].audio
            )
        else:
            track_audio = track.audio

        print(track_name)

        augmented_gain = None

        # if args.loudnorm_input_lufs:  # If you want to use loud-normalized input
        track_lufs = meter.integrated_loudness(track_audio)
        augmented_gain = args.loudnorm_input_lufs - track_lufs
        if os.path.basename(args.data_root) == "musdb18hq":
            if args.target != "mixture":
                track_audio = track.targets[args.target].audio
            track_audio = track_audio * db2linear(augmented_gain, eps=0.0)
        elif os.path.basename(args.data_root) == "musdb_XL":
            track_audio = track_audio * db2linear(augmented_gain, eps=0.0)
            if args.target != "mixture":
                # stem_audio = track.targets[args.target].audio # We are not going to use this to avoid some clipping errors.
                hq_track = hq_tracks[idx]
                hq_audio = hq_track.audio
                hq_stem = hq_track.targets[args.target].audio
                samplewise_gain = track_audio / (hq_audio + 1e-8)
                track_audio = samplewise_gain * hq_stem

        os.makedirs(f"{args.output_directory}/{track_name}", exist_ok=True)
        sf.write(
            f"{args.output_directory}/{track_name}/{args.target}.wav", track_audio, 44100
        )
        # if os.path.basename(args.data_root) == "musdb18hq"
        #   pass
        # elif os.path.basename(args.data_root) == "musdb_XL":
        #   (loudnorm_XL / musdb_hq) * hq_stem  => loudnorm_XL_stem

        if args.save_16k_mono:
            track_audio_16k_mono = librosa.to_mono(track_audio.T)
            track_audio_16k_mono = librosa.resample(
                track_audio_16k_mono,
                orig_sr=44100,
                target_sr=16000,
            )
            os.makedirs(f"{args.output_directory}_16k_mono/{track_name}", exist_ok=True)
            sf.write(
                f"{args.output_directory}_16k_mono/{track_name}/{args.target}.wav",
                track_audio_16k_mono,
                samplerate=16000,
            )


if __name__ == "__main__":
    main()
