# Calculate SI-SDR, Multi-resolution spectrogram mse score of the pre-inferenced sources
import os
import argparse
import csv
import json
import glob

import tqdm
import numpy as np
import librosa
import musdb
import pyloudnorm as pyln

from utils import str2bool, db2linear

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
    default="/data2/personal/jeon/delimit/data/musdb_hq_loudnorm",
)
parser.add_argument(
    "--output_directory",
    type=str,
    default="/data2/personal/jeon/delimit/results",
)
parser.add_argument("--exp_name", type=str, default="convtasnet_35")
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


est_track_list = glob.glob(f"{args.test_output_dir}/*/{args.target}.wav")
f = open(
    f"{args.test_output_dir}/score_feature_{args.target}.json",
    encoding="UTF-8",
)
dict_song_score_est = json.loads(f.read())
# f"{args.test_output_dir}/score_feature_{args.target}.json"

if args.target == "all":
    ref_track_list = glob.glob(f"{args.root}/*/mixture.wav")
    # dict_song_score_ref = json.loads(f"{args.root}/score_feature.json")
    f = open(f"{args.root}/score_feature.json", encoding="UTF-8")
    dict_song_score_ref = json.loads(f.read())
else:
    # ref_track_list = musdb.DB(root=args.root, subsets="test", is_wav=True)
    ref_track_list = glob.glob(f"{args.root}/*/{args.target}.wav")
    # dict_song_score_ref = json.loads(f"{args.root}/score_feature_{args.target}.json")
    f = open(f"{args.root}/score_feature_{args.target}.json", encoding="UTF-8")
    dict_song_score_ref = json.loads(f.read())

i = 0

dict_song_score = {}
list_diff_dynamic_complexity = []

for track in tqdm.tqdm(ref_track_list):
    audio_name = os.path.basename(os.path.dirname(track))
    ref_dyn_complexity = dict_song_score_ref[audio_name]["dynamic_complexity_score"]
    est_dyn_complexity = dict_song_score_est[audio_name]["dynamic_complexity_score"]

    list_diff_dynamic_complexity.append(est_dyn_complexity - ref_dyn_complexity)

    i += 1

print(
    f"Dynamic complexity difference {args.exp_name} vs {os.path.basename(args.root)} on {args.target}"
)
print("mean: ", np.mean(list_diff_dynamic_complexity))
print("median: ", np.median(list_diff_dynamic_complexity))
print("std: ", np.std(list_diff_dynamic_complexity))
# print(list_diff_dynamic_complexity)
