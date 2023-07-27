import os
import argparse
import csv
import json
import glob
from typing import Any, Optional, Union, Collection

import tqdm
import numpy as np
import librosa
from librosa.core.spectrum import _spectrogram
import musdb
import essentia
import essentia.standard
import pyloudnorm as pyln

from utils import str2bool, db2linear


def spectral_crest(
    *,
    y: Optional[np.ndarray] = None,
    S: Optional[np.ndarray] = None,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: Optional[int] = None,
    window: str = "hann",
    center: bool = True,
    pad_mode: str = "constant",
    amin: float = 1e-10,
    power: float = 2.0,
) -> np.ndarray:
    """Compute spectral crest

    Spectral crest (or tonality coefficient) is a measure of
    the ratio of the maximum of the spectrum to the arithmetic mean of the spectrum

    A higher spectral crest => more tonality,
    A lower spectral crest => more noisy.


    Parameters
    ----------
    y : np.ndarray [shape=(..., n)] or None
        audio time series. Multi-channel is supported.
    S : np.ndarray [shape=(..., d, t)] or None
        (optional) pre-computed spectrogram magnitude
    n_fft : int > 0 [scalar]
        FFT window size
    hop_length : int > 0 [scalar]
        hop length for STFT. See `librosa.stft` for details.
    win_length : int <= n_fft [scalar]
        Each frame of audio is windowed by `window()`.
        The window will be of length `win_length` and then padded
        with zeros to match ``n_fft``.
        If unspecified, defaults to ``win_length = n_fft``.
    window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
        - a window specification (string, tuple, or number);
          see `scipy.signal.get_window`
        - a window function, such as `scipy.signal.windows.hann`
        - a vector or array of length ``n_fft``
        .. see also:: `librosa.filters.get_window`
    center : boolean
        - If `True`, the signal ``y`` is padded so that frame
          ``t`` is centered at ``y[t * hop_length]``.
        - If `False`, then frame `t` begins at ``y[t * hop_length]``
    pad_mode : string
        If ``center=True``, the padding mode to use at the edges of the signal.
        By default, STFT uses zero padding.
    amin : float > 0 [scalar]
        minimum threshold for ``S`` (=added noise floor for numerical stability)
    power : float > 0 [scalar]
        Exponent for the magnitude spectrogram.
        e.g., 1 for energy, 2 for power, etc.
        Power spectrogram is usually used for computing spectral flatness.

    Returns
    -------
    crest : np.ndarray [shape=(..., 1, t)]
        spectral crest for each frame.


    """

    S, n_fft = _spectrogram(
        y=y,
        S=S,
        n_fft=n_fft,
        hop_length=hop_length,
        power=1.0,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
    )

    S_thresh = np.maximum(amin, S**power)
    # gmean = np.exp(np.mean(np.log(S_thresh), axis=-2, keepdims=True))
    gmax = np.max(S_thresh, axis=-2, keepdims=True)
    amean = np.mean(S_thresh, axis=-2, keepdims=True)
    crest: np.ndarray = gmax / amean
    return crest


parser = argparse.ArgumentParser(description="model test.py")

parser.add_argument(
    "--target",
    type=str,
    default="all",
    help="target source. all, vocals, drums, bass, other",
)
parser.add_argument(
    "--root", type=str, default="/path/to/musdb18hq_loudnorm"
)
parser.add_argument("--exp_name", type=str, default="delimit_6_s")
parser.add_argument(
    "--output_directory",
    type=str,
    default="/path/to/results",
)
parser.add_argument(
    "--calc_results",
    type=str2bool,
    default=True,
    help="calculate results or musdb-hq or musdb-XL test dataset",
)


args, _ = parser.parse_known_args()

args.sample_rate = 44100

args.test_output_dir = f"{args.output_directory}/test/{args.exp_name}"

if args.calc_results:
    track_list = glob.glob(
        f"{args.output_directory}/test/{args.exp_name}/*/{args.target}.wav"
    )
else:
    if args.target == "all":
        track_list = glob.glob(f"{args.root}/*/mixture.wav")
    else:
        track_list = glob.glob(f"{args.root}/*/{args.target}.wav")

i = 0


dynamic_complexity = essentia.standard.DynamicComplexity()
loudness_range = essentia.standard.LoudnessEBUR128()
spectral_centroid = essentia.standard.SpectralCentroidTime()
crest = essentia.standard.Crest()
dynamic_spread = essentia.standard.DistributionShape()
central_moments = essentia.standard.CentralMoments()

dict_song_score = {}
list_rms = []
list_crest_factor = []
list_dc_score = []
list_lra_score = []
list_sc_hertz = []
list_sf_score = []
list_spectral_crest_score = []

for track in tqdm.tqdm(track_list):
    audio_name = os.path.basename(os.path.dirname(track))
    gt_source_librosa = librosa.load(f"{track}", sr=args.sample_rate, mono=False)[
        0
    ]  # (nb_channels, nb_samples)
    gt_source_librosa_mono = librosa.to_mono(gt_source_librosa)  # (nb_samples)

    gt_source_essentia = essentia.standard.AudioLoader(filename=f"{track}")()[
        0
    ]  # (nb_samples, nb_channels)
    gt_source_essentia_cat = np.concatenate(
        [gt_source_essentia[:, 0], gt_source_essentia[:, 1]]
    )  # (nb_samples * nb_channels)
    gt_source_essentia_mono = np.mean(gt_source_essentia, axis=1)  # (nb_samples)

    rms = np.sqrt(np.mean(gt_source_essentia_cat**2))
    crest_factor = np.max(np.abs(gt_source_essentia_cat)) / rms

    dc_score, _ = dynamic_complexity(gt_source_essentia_mono)
    _, _, _, lra_score = loudness_range(gt_source_essentia)
    sc_hertz = spectral_centroid(gt_source_essentia_mono)
    sf_score = np.mean(librosa.feature.spectral_flatness(gt_source_librosa_mono))
    spectral_crest_score = np.mean(spectral_crest(y=gt_source_librosa_mono))

    dict_song_score[audio_name] = {
        "rms": float(rms),
        "crest_factor": float(crest_factor),
        "dynamic_complexity_score": float(dc_score),
        "lra_score": float(lra_score),
        "spectral_centroid_hertz": float(sc_hertz),
        "spectral_flatness_score": float(sf_score),
        "spectral_crest_score": float(spectral_crest_score),
    }
    list_rms.append(rms)
    list_crest_factor.append(crest_factor)
    list_dc_score.append(dc_score)
    list_lra_score.append(lra_score)
    list_sc_hertz.append(sc_hertz)
    list_sf_score.append(sf_score)
    list_spectral_crest_score.append(spectral_crest_score)

    i += 1

if args.calc_results:
    print(f"{args.exp_name} on {args.target}")
else:
    print(f"{os.path.basename(args.root)} on {args.target}")
print(f"rms: {np.mean(list_rms)}")
print(f"crest_factor: {np.mean(list_crest_factor)}")
print(f"dynamic_complexity_score: {np.mean(list_dc_score)}")
print(f"lra_score: {np.mean(list_lra_score)}")
print(f"sc_hertz: {np.mean(list_sc_hertz)}")
print(f"sf_score: {np.mean(list_sf_score)}")
print(f"spectral_crest_score: {np.mean(list_spectral_crest_score)}")


# save dict_song_score to json file
if args.target == "all":
    file_name = "score_features"
else:
    file_name = f"score_feature_{args.target}"
if args.calc_results:
    with open(
        f"{args.output_directory}/test/{args.exp_name}/{file_name}.json", "w"
    ) as f:
        json.dump(dict_song_score, f, indent=4)
else:
    with open(f"{args.root}/{file_name}.json", "w") as f:
        json.dump(dict_song_score, f, indent=4)
