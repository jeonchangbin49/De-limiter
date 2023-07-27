# We are going to use PEAQ based on https://github.com/HSU-ANT/gstpeaq

"""
python3 score_peaq.py --exp_name=convtasnet_35 | tee /home/jeon/results/delimit/test/convtasnet_35/score_peaq.txt
"""

"""
python3 score_peaq.py --exp_name=convtasnet_35_demucs --target=vocals --root=/home/jeon/results/delimit/test/musdb_XL_loudnorm_demucs | tee /home/jeon/results/delimit/test/convtasnet_35_demucs/score_peaq_vocals.txt
"""

"""
python3 score_peaq.py --exp_name=musdb_XL_loudnorm --target=mixture --output_directory=/home/jeon/data/musdb_related --calc_results=False | tee /home/jeon/data/musdb_related/musdb_XL_loudnorm/score_peaq_mixture.txt
"""


import os
import subprocess
import glob
import argparse


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


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
    default="/home/jeon/data/musdb_related/musdb_XL_loudnorm",
)
parser.add_argument(
    "--output_directory",
    type=str,
    default="/home/jeon/results/delimit",
)
parser.add_argument("--exp_name", type=str, default="convtasnet_35")
parser.add_argument(
    "--calc_results",
    type=str2bool,
    default=True,
    help="Set this True when you want to calculate the results of the test set. Set this False when calculating musdb-hq vs musdb-XL. (top row in Table 1.)",
)

args, _ = parser.parse_known_args()

if args.calc_results:
    args.test_output_dir = f"{args.output_directory}/test/{args.exp_name}"
else:
    args.test_output_dir = f"{args.output_directory}/{args.exp_name}"

if args.target == "all":
    song_list = sorted(glob.glob(f"{args.root}/*/mixture.wav"))

    for song in song_list:
        song_name = os.path.basename(os.path.dirname(song))
        est_path = f"{args.test_output_dir}/{song_name}/{args.target}.wav"
        subprocess.run(
            f'peaq --gst-plugin-load=/usr/local/lib/gstreamer-1.0/libgstpeaq.so "{song}" "{est_path}"',
            shell=True,
        )

else:
    song_list = sorted(glob.glob(f"{args.root}/*/{args.target}.wav"))

    for song in song_list:
        song_name = os.path.basename(os.path.dirname(song))
        est_path = f"{args.test_output_dir}/{song_name}/{args.target}.wav"
        subprocess.run(
            f'peaq --gst-plugin-load=/usr/local/lib/gstreamer-1.0/libgstpeaq.so "{song}" "{est_path}"',
            shell=True,
        )
