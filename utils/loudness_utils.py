import random

import numpy as np
import torch


def linear2db(x, eps=1e-5, scale=20):
    return scale * np.log10(x + eps)


def db2linear(x, eps=1e-5, scale=20):
    return 10 ** (x / scale) - eps


def normalize_mag_spec(S, min_level_db=-100.0):
    return torch.clamp((S - min_level_db) / -min_level_db, min=0.0, max=1.0)


def denormalize_mag_spec(S, min_level_db=-100.0):
    return torch.clamp(S, min=0.0, max=1.0) * -min_level_db + min_level_db


def loudness_match_and_norm(audio1, audio2, meter):
    lufs_1 = meter.integrated_loudness(audio1)
    lufs_2 = meter.integrated_loudness(audio2)

    if np.isinf(lufs_1) or np.isinf(lufs_2):
        return audio1, audio2
    else:
        audio2 = audio2 * db2linear(lufs_1 - lufs_2)

        return audio1, audio2


def loudness_normal_match_and_norm(audio1, audio2, meter):
    lufs_1 = meter.integrated_loudness(audio1)
    lufs_2 = meter.integrated_loudness(audio2)

    if np.isinf(lufs_1) or np.isinf(lufs_2):
        return audio1, audio2
    else:
        target_lufs = random.normalvariate(lufs_1, 6.0)
        audio2 = audio2 * db2linear(target_lufs - lufs_2)

        return audio1, audio2


def loudness_normal_match_and_norm_output_louder_first(audio1, audio2, meter):
    lufs_1 = meter.integrated_loudness(audio1)
    lufs_2 = meter.integrated_loudness(audio2)

    if np.isinf(lufs_1) or np.isinf(lufs_2):
        return audio1, audio2
    else:
        target_lufs = random.normalvariate(
            lufs_1 - 2.0, 2.0
        )  # we want audio1 to be louder than audio2 about target_lufs_diff
        audio2 = audio2 * db2linear(target_lufs - lufs_2)

        return audio1, audio2


def loudnorm(audio, target_lufs, meter, eps=1e-5):
    lufs = meter.integrated_loudness(audio)
    if np.isinf(lufs):
        return audio, 0.0
    else:
        adjusted_gain = target_lufs - lufs
        audio = audio * db2linear(adjusted_gain, eps)

        return audio, adjusted_gain
