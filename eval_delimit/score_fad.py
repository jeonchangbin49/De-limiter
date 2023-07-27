# We are going to use FAD based on https://github.com/gudgud96/frechet-audio-distance
import os
import subprocess
import glob
import argparse

from frechet_audio_distance import FrechetAudioDistance

from utils import str2bool


parser = argparse.ArgumentParser(description="model test.py")

parser.add_argument(
    "--target",
    type=str,
    default="all",
    help="target source. all, vocals, drums, bass, other",
)
parser.add_argument(
    "--root",
    type=str,
    default="/path/to/musdb18hq_loudnorm",
)
parser.add_argument(
    "--output_directory",
    type=str,
    default="/path/to/results",
)
parser.add_argument("--exp_name", type=str, default="delimit_6_s")
parser.add_argument(
    "--calc_results",
    type=str2bool,
    default=True,
    help="Set this True when you want to calculate the results of the test set. Set this False when calculating musdb-hq vs musdb-XL. (top row in Table 1.)",
)

args, _ = parser.parse_known_args()

os.makedirs(f"{args.root}/musdb_hq_loudnorm_16k_mono_link", exist_ok=True)

song_list = glob.glob(f"{args.root}/musdb_hq_loudnorm_16k_mono/*/mixture.wav")
for song in song_list:
    song_name = os.path.basename(os.path.dirname(song))
    subprocess.run(
        f'ln --symbolic "{song}" "{args.root}/musdb_hq_loudnorm_16k_mono_link/{song_name}.wav"',
        shell=True,
    )


if args.calc_results:
    args.test_output_dir = f"{args.output_directory}/test/{args.exp_name}"
else:
    args.test_output_dir = f"{args.output_directory}/{args.exp_name}"

os.makedirs(f"{args.test_output_dir}_16k_mono_link", exist_ok=True)

song_list = glob.glob(f"{args.test_output_dir}_16k_mono/*/{args.target}.wav")
for song in song_list:
    song_name = os.path.basename(os.path.dirname(song))
    subprocess.run(
        f'ln --symbolic "{song}" "{args.test_output_dir}_16k_mono_link/{song_name}.wav"',
        shell=True,
    )


frechet = FrechetAudioDistance()

fad_score = frechet.score(
    f"{args.root}/musdb_hq_loudnorm_16k_mono_link",
    f"{args.test_output_dir}_16k_mono_link",
)

print(f"{args.exp_name}")
print(f"FAD score: {fad_score}")
