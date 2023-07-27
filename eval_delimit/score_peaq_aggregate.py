# PEAQ aggregate score
"""
/home/jeon/results/delimit/test/convtasnet_35/score_peaq.txt
"""

import os
import glob
import argparse
import json


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
    default="/home/jeon/data/musdb_related",
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
    score_path = f"{args.test_output_dir}/score_peaq.txt"
else:
    score_path = f"{args.test_output_dir}/score_peaq_{args.target}.txt"

# write the code to load score_peaq.txt
with open(score_path, "r") as f:
    score_txt = f.readlines()

song_list = glob.glob(f"{args.root}/musdb_hq_loudnorm/*")

dict_song_peaq = {}
list_peaq = []
for idx, song in enumerate(song_list):
    song_name = os.path.basename(song)
    peaq = float(score_txt[idx * 2].replace("Objective Difference Grade: ", ""))
    dict_song_peaq[song_name] = peaq
    list_peaq.append(peaq)

print(f"{args.exp_name} on {args.target}")
print(f"PEAQ score: {sum(list_peaq) / len(list_peaq)}")

if args.target == "all":
    # save dict_song_peaq to json file
    with open(f"{args.test_output_dir}/score_peaq.json", "w") as f:
        json.dump(dict_song_peaq, f, indent=4)
else:
    # save dict_song_peaq to json file
    with open(
        f"{args.test_output_dir}/score_peaq_{args.target}.json",
        "w",
    ) as f:
        json.dump(dict_song_peaq, f, indent=4)
