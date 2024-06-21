import os
import random
from typing import Optional, Callable
import json
import glob
import csv

import numpy as np
import torch
import librosa
import pyloudnorm as pyln
from pedalboard import Pedalboard, Limiter, Gain, Compressor, Clipping

from .dataset import (
    MusdbTrainDataset,
    MusdbValidDataset,
    apply_limitaug,
)
from utils import (
    load_wav_arbitrary_position_stereo,
    load_wav_specific_position_stereo,
    db2linear,
    str2bool,
)


class DelimitTrainDataset(MusdbTrainDataset):
    def __init__(
        self,
        target: str = "all",
        root: str = None,
        seq_duration: Optional[float] = 6.0,
        samples_per_track: int = 64,
        source_augmentations: Optional[Callable] = lambda audio: audio,
        sample_rate: int = 44100,
        seed: int = 42,
        limitaug_method: str = "limitaug",
        limitaug_mode: str = "normal_L",
        limitaug_custom_target_lufs: float = None,
        limitaug_custom_target_lufs_std: float = None,
        target_loudnorm_lufs: float = -14.0,
        target_limitaug_mode: str = None,
        target_limitaug_custom_target_lufs: float = None,
        target_limitaug_custom_target_lufs_std: float = None,
        custom_limiter_attack_range: list = [2.0, 2.0],
        custom_limiter_release_range: list = [200.0, 200.0],
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            target=target,
            root=root,
            seq_duration=seq_duration,
            samples_per_track=samples_per_track,
            source_augmentations=source_augmentations,
            sample_rate=sample_rate,
            seed=seed,
            limitaug_method=limitaug_method,
            limitaug_mode=limitaug_mode,
            limitaug_custom_target_lufs=limitaug_custom_target_lufs,
            limitaug_custom_target_lufs_std=limitaug_custom_target_lufs_std,
            target_loudnorm_lufs=target_loudnorm_lufs,
            custom_limiter_attack_range=custom_limiter_attack_range,
            custom_limiter_release_range=custom_limiter_release_range,
            *args,
            **kwargs,
        )

        self.target_limitaug_mode = target_limitaug_mode

        self.target_limitaug_custom_target_lufs = (target_limitaug_custom_target_lufs,)
        self.target_limitaug_custom_target_lufs_std = (
            target_limitaug_custom_target_lufs_std,
        )
        self.limitaug_mode_statistics["target_custom"] = [
            target_limitaug_custom_target_lufs,
            target_limitaug_custom_target_lufs_std,
        ]

        """
        Parameters
        ----------
        limitaug_method : str
        choose from ["linear_gain_increase", "limitaug", "limitaug_then_loudnorm", "only_loudnorm"]
        limitaug_mode : str
        choose from ["uniform", "normal", "normal_L", "normal_XL", "normal_short_term", "normal_L_short_term", "normal_XL_short_term", "custom"]
        limitaug_custom_target_lufs : float
        valid only when
        limitaug_mode == "custom"
        target_loudnorm_lufs : float
        valid only when
        limitaug_method == 'limitaug_then_loudnorm' or 'only_loudnorm'
        default is -14.
        To the best of my knowledge, Spotify and Youtube music is using -14 as a reference loudness normalization level.
        No special reason for the choice of -14 as target_loudnorm_lufs.
        target : str
            target name of the source to be separated, defaults to ``vocals``.
        root : str
            root path of MUSDB
        seq_duration : float
            training is performed in chunks of ``seq_duration`` (in seconds,
            defaults to ``None`` which loads the full audio track
        samples_per_track : int
            sets the number of samples, yielded from each track per epoch.
            Defaults to 64
        source_augmentations : list[callables]
            provide list of augmentation function that take a multi-channel
            audio file of shape (src, samples) as input and output. Defaults to
            no-augmentations (input = output)
        seed : int
            control randomness of dataset iterations
        args, kwargs : additional keyword arguments
            used to add further control for the musdb dataset
            initialization function.
        """

    # Get a limitaug result without target (individual stem source)
    def get_limitaug_mixture(self, mixture):
        if self.limitaug_method == "limitaug":
            self.board[1].release_ms = random.uniform(30.0, 200.0)
            target_lufs = self.sample_target_lufs()
            mixture_limited, mixture_lufs = apply_limitaug(
                mixture,
                self.board,
                self.meter,
                self.sample_rate,
                target_lufs=target_lufs,
            )

        elif self.limitaug_method == "limitaug_then_loudnorm":
            self.board[1].release_ms = random.uniform(30.0, 200.0)
            target_lufs = self.sample_target_lufs()
            mixture_limited, mixture_lufs = (
                apply_limitaug(
                    mixture,
                    self.board,
                    self.meter,
                    self.sample_rate,
                    target_lufs=target_lufs,
                    target_loudnorm_lufs=self.target_loudnorm_lufs,
                ),
            )

        # Apply LimitAug using Custom Limiter
        elif self.limitaug_method == "custom_limiter_limitaug":
            # Change attack time of First compressor of the Limiter
            self.board[1].attack_ms = random.uniform(
                self.custom_limiter_attack_range[0], self.custom_limiter_attack_range[1]
            )
            # Change release time of First compressor of the Limiter
            self.board[1].release_ms = random.uniform(
                self.custom_limiter_release_range[0],
                self.custom_limiter_release_range[1],
            )
            # Change release time of Second compressor of the Limiter
            self.board[2].release_ms = random.uniform(30.0, 200.0)
            target_lufs = self.sample_target_lufs()
            mixture_limited, mixture_lufs = apply_limitaug(
                mixture,
                self.board,
                self.meter,
                self.sample_rate,
                target_lufs=target_lufs,
                target_loudnorm_lufs=self.target_loudnorm_lufs,
            )

        # When we want to force NN to output an appropriately compressed target output
        if self.target_limitaug_mode:
            mixture_target_lufs = random.gauss(
                self.limitaug_mode_statistics[self.target_limitaug_mode][0],
                self.limitaug_mode_statistics[self.target_limitaug_mode][1],
            )
            mixture, target_lufs = apply_limitaug(
                mixture,
                self.board,
                self.meter,
                self.sample_rate,
                target_lufs=mixture_target_lufs,
                loudness=mixture_lufs,
            )

        if np.isinf(mixture_lufs):
            mixture_loudnorm = mixture
        else:
            augmented_gain = self.target_loudnorm_lufs - mixture_lufs
            mixture_loudnorm = mixture * db2linear(augmented_gain, eps=0.0)

        return mixture_limited, mixture_loudnorm

    def __getitem__(self, index):
        audio_sources = []

        for k, source in enumerate(self.sources):
            # memorize index of target source
            if source == self.target:  # if source is 'vocals'
                track_path = self.train_list[
                    index // self.samples_per_track
                ]  # we want to use # training samples per each track.
                audio_path = f"{track_path}/{source}.wav"
                audio = load_wav_arbitrary_position_stereo(
                    audio_path, self.sample_rate, self.seq_duration
                )
            else:
                track_path = random.choice(self.train_list)
                audio_path = f"{track_path}/{source}.wav"
                audio = load_wav_arbitrary_position_stereo(
                    audio_path, self.sample_rate, self.seq_duration
                )
            audio = self.source_augmentations(audio)
            audio_sources.append(audio)

        stems = np.stack(audio_sources, axis=0)

        # apply linear mix over source index=0
        # and here, linear mixture is a target unlike in MusdbTrainDataset
        mixture = stems.sum(0)
        mixture_limited, mixture_loudnorm = self.get_limitaug_mixture(mixture)
        # We will give mixture_limited as an input and mixture_loudnorm as a target to the model.

        mixture_limited = np.clip(mixture_limited, -1.0, 1.0)
        mixture_limited = torch.as_tensor(mixture_limited, dtype=torch.float32)
        mixture_loudnorm = torch.as_tensor(mixture_loudnorm, dtype=torch.float32)

        return mixture_limited, mixture_loudnorm


class OzoneTrainDataset(DelimitTrainDataset):
    def __init__(
        self,
        target: str = "all",
        root: str = None,
        ozone_root: str = None,
        use_fixed: float = 0.1,  # ratio of fixed samples
        seq_duration: Optional[float] = 6.0,
        samples_per_track: int = 64,
        source_augmentations: Optional[Callable] = lambda audio: audio,
        sample_rate: int = 44100,
        seed: int = 42,
        limitaug_method: str = "limitaug",
        limitaug_mode: str = "normal_L",
        limitaug_custom_target_lufs: float = None,
        limitaug_custom_target_lufs_std: float = None,
        target_loudnorm_lufs: float = -14.0,
        target_limitaug_mode: str = None,
        target_limitaug_custom_target_lufs: float = None,
        target_limitaug_custom_target_lufs_std: float = None,
        custom_limiter_attack_range: list = [2.0, 2.0],
        custom_limiter_release_range: list = [200.0, 200.0],
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            target,
            root,
            seq_duration,
            samples_per_track,
            source_augmentations,
            sample_rate,
            seed,
            limitaug_method,
            limitaug_mode,
            limitaug_custom_target_lufs,
            limitaug_custom_target_lufs_std,
            target_loudnorm_lufs,
            target_limitaug_mode,
            target_limitaug_custom_target_lufs,
            target_limitaug_custom_target_lufs_std,
            custom_limiter_attack_range,
            custom_limiter_release_range,
            *args,
            **kwargs,
        )

        self.ozone_root = ozone_root
        self.use_fixed = use_fixed
        self.list_train_fixed = glob.glob(f"{self.ozone_root}/ozone_train_fixed/*.wav")
        self.list_dict_train_random = []

        # Load information of pre-generated random training examples
        list_csv_files = glob.glob(f"{self.ozone_root}/ozone_train_random_*.csv")
        list_csv_files.sort()
        for csv_file in list_csv_files:
            with open(csv_file, "r") as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    self.list_dict_train_random.append(
                        {
                            row[0]: {
                                "max_threshold": float(row[1]),
                                "max_character": float(row[2]),
                                "vocals": {
                                    "name": row[3],
                                    "start_sec": float(row[4]),
                                    "gain": float(row[5]),
                                    "channelswap": str2bool(row[6]),
                                },
                                "bass": {
                                    "name": row[7],
                                    "start_sec": float(row[8]),
                                    "gain": float(row[9]),
                                    "channelswap": str2bool(row[10]),
                                },
                                "drums": {
                                    "name": row[11],
                                    "start_sec": float(row[12]),
                                    "gain": float(row[13]),
                                    "channelswap": str2bool(row[14]),
                                },
                                "other": {
                                    "name": row[15],
                                    "start_sec": float(row[16]),
                                    "gain": float(row[17]),
                                    "channelswap": str2bool(row[18]),
                                },
                            }
                        }
                    )

    def __getitem__(self, idx):
        use_fixed_prob = random.random()

        if use_fixed_prob <= self.use_fixed:
            # Fixed examples
            audio_path = random.choice(self.list_train_fixed)
            song_name = os.path.basename(audio_path).replace(".wav", "")
            mixture_limited, start_pos_sec = load_wav_arbitrary_position_stereo(
                audio_path, self.sample_rate, self.seq_duration, return_pos=True
            )

            audio_sources = []
            track_path = f"{self.root}/train/{song_name}"
            for source in self.sources:
                audio_path = f"{track_path}/{source}.wav"
                audio = load_wav_specific_position_stereo(
                    audio_path,
                    self.sample_rate,
                    self.seq_duration,
                    start_position=start_pos_sec,
                )
                audio_sources.append(audio)

        else:
            # Random examples
            # Load mixture_limited (pre-generated)
            dict_seg = random.choice(self.list_dict_train_random)
            seg_name = list(dict_seg.keys())[0]
            audio_path = f"{self.ozone_root}/ozone_train_random/{seg_name}.wav"
            dict_seg_info = dict_seg[seg_name]

            mixture_limited, sr = librosa.load(
                audio_path, sr=self.sample_rate, mono=False
            )

            # Load mixture_unlimited (from the original musdb18, using metadata)
            audio_sources = []

            for source in self.sources:
                dict_seg_source_info = dict_seg_info[source]
                audio_path = (
                    f"{self.root}/train/{dict_seg_source_info['name']}/{source}.wav"
                )
                audio = load_wav_specific_position_stereo(
                    audio_path,
                    self.sample_rate,
                    self.seq_duration,
                    start_position=dict_seg_source_info["start_sec"],
                )

                # apply augmentations
                audio = audio * dict_seg_source_info["gain"]
                if dict_seg_source_info["channelswap"]:
                    audio = np.flip(audio, axis=0)

                audio_sources.append(audio)

        stems = np.stack(audio_sources, axis=0)
        mixture = stems.sum(axis=0)
        mixture_lufs = self.meter.integrated_loudness(mixture.T)
        if np.isinf(mixture_lufs):
            mixture_loudnorm = mixture
        else:
            augmented_gain = self.target_loudnorm_lufs - mixture_lufs
            mixture_loudnorm = mixture * db2linear(augmented_gain, eps=0.0)

        return mixture_limited, mixture_loudnorm


class DelimitValidDataset(MusdbValidDataset):
    def __init__(
        self,
        target: str = "vocals",
        root: str = None,
        delimit_valid_root: str = None,
        valid_target_lufs: float = -8.05,  # From the Table 1 of the "Towards robust music source separation on loud commercial music" paper, the average loudness of commerical music.
        target_loudnorm_lufs: float = -14.0,
        delimit_valid_L_root: str = None,  # This will be used when using the target as compressed (normal_L) mixture.
        use_custom_limiter: bool = False,
        custom_limiter_attack_range: list = [0.1, 10.0],
        custom_limiter_release_range: list = [30.0, 200.0],
        *args,
        **kwargs,
    ) -> None:
        super().__init__(target=target, root=root, *args, **kwargs)
        self.delimit_valid_root = delimit_valid_root
        if self.delimit_valid_root:
            with open(f"{self.delimit_valid_root}/valid_loudness.json", "r") as f:
                self.dict_valid_loudness = json.load(f)
        self.delimit_valid_L_root = delimit_valid_L_root
        if self.delimit_valid_L_root:
            with open(f"{self.delimit_valid_L_root}/valid_loudness.json", "r") as f:
                self.dict_valid_L_loudness = json.load(f)

        self.valid_target_lufs = valid_target_lufs
        self.target_loudnorm_lufs = target_loudnorm_lufs
        self.meter = pyln.Meter(self.sample_rate)
        self.use_custom_limiter = use_custom_limiter

        if self.use_custom_limiter:
            print("using Custom limiter limitaug for validation!!")
            self.custom_limiter_attack_range = custom_limiter_attack_range
            self.custom_limiter_release_range = custom_limiter_release_range
            self.board = Pedalboard(
                [
                    Gain(gain_db=0.0),
                    Compressor(
                        threshold_db=-10.0, ratio=4.0, attack_ms=2.0, release_ms=200.0
                    ),  # attack_ms and release_ms will be changed later.
                    Compressor(
                        threshold_db=0.0,
                        ratio=1000.0,
                        attack_ms=0.001,
                        release_ms=100.0,
                    ),
                    Gain(gain_db=3.75),
                    Clipping(threshold_db=0.0),
                ]
            )  # This implementation is the same as JUCE Limiter.
            # However, we want the first compressor to have a variable attack and release time.
            # Therefore, we use the Custom Limiter instead of the JUCE Limiter.
        else:
            self.board = Pedalboard(
                [Gain(gain_db=0.0), Limiter(threshold_db=0.0, release_ms=100.0)]
            )  # Currently, we are using a limiter with a release time of 100ms.

    def __getitem__(self, index):
        audio_sources = []
        target_ind = None

        for k, source in enumerate(self.sources):
            # memorize index of target source
            if source == self.target:  # if source is 'vocals'
                target_ind = k
                track_path = self.valid_list[index]
                song_name = os.path.basename(track_path)
                audio_path = f"{track_path}/{source}.wav"
                # audio = utils.load_wav_stereo(audio_path, self.sample_rate)
                audio = librosa.load(audio_path, mono=False, sr=self.sample_rate)[0]
            else:
                track_path = self.valid_list[index]
                song_name = os.path.basename(track_path)
                audio_path = f"{track_path}/{source}.wav"
                # audio = utils.load_wav_stereo(audio_path, self.sample_rate)
                audio = librosa.load(audio_path, mono=False, sr=self.sample_rate)[0]

            audio = torch.as_tensor(audio, dtype=torch.float32)
            audio_sources.append(audio)

        stems = np.stack(audio_sources, axis=0)

        # apply linear mix over source index=0
        # and here, linear mixture is a target unlike in MusdbTrainDataset
        mixture = stems.sum(0)
        if (
            self.delimit_valid_root
        ):  # If there exists a pre-processed delimit valid dataset
            audio_path = f"{self.delimit_valid_root}/valid/{song_name}.wav"
            mixture_limited = librosa.load(audio_path, mono=False, sr=self.sample_rate)[
                0
            ]
            mixture_lufs = self.dict_valid_loudness[song_name]

        else:
            if self.use_custom_limiter:
                custom_limiter_attack = random.uniform(
                    self.custom_limiter_attack_range[0],
                    self.custom_limiter_attack_range[1],
                )
                self.board[1].attack_ms = custom_limiter_attack

                custom_limiter_release = random.uniform(
                    self.custom_limiter_release_range[0],
                    self.custom_limiter_release_range[1],
                )
                self.board[1].release_ms = custom_limiter_release

                mixture_limited, mixture_lufs = apply_limitaug(
                    mixture,
                    self.board,
                    self.meter,
                    self.sample_rate,
                    target_lufs=self.valid_target_lufs,
                )
            else:
                mixture_limited, mixture_lufs = apply_limitaug(
                    mixture,
                    self.board,
                    self.meter,
                    self.sample_rate,
                    target_lufs=self.valid_target_lufs,
                    # target_loudnorm_lufs=self.target_loudnorm_lufs,
                )  # mixture_limited is a limiter applied mixture
                # We will give mixture_limited as an input and mixture_loudnorm as a target to the model.

        if self.delimit_valid_L_root:
            audio_L_path = f"{self.delimit_valid_L_root}/valid/{song_name}.wav"
            mixture_loudnorm = librosa.load(
                audio_L_path, mono=False, sr=self.sample_rate
            )[0]
            mixture_lufs = self.dict_valid_L_loudness[song_name]
            mixture = mixture_loudnorm

        augmented_gain = self.target_loudnorm_lufs - mixture_lufs
        mixture_loudnorm = mixture * db2linear(augmented_gain)

        if self.use_custom_limiter:
            return (
                mixture_limited,
                mixture_loudnorm,
                song_name,
                mixture_lufs,
                custom_limiter_attack,
                custom_limiter_release,
            )
        else:
            return mixture_limited, mixture_loudnorm, song_name, mixture_lufs


class OzoneValidDataset(MusdbValidDataset):
    def __init__(
        self,
        target: str = "all",
        root: str = None,
        ozone_root: str = None,
        target_loudnorm_lufs: float = -14.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(target=target, root=root, *args, **kwargs)

        self.ozone_root = ozone_root
        self.target_loudnorm_lufs = target_loudnorm_lufs

        with open(f"{self.ozone_root}/valid_loudness.json", "r") as f:
            self.dict_valid_loudness = json.load(f)

    def __getitem__(self, index):
        audio_sources = []

        track_path = self.valid_list[index]
        song_name = os.path.basename(track_path)
        for k, source in enumerate(self.sources):
            audio_path = f"{track_path}/{source}.wav"
            audio = librosa.load(audio_path, mono=False, sr=self.sample_rate)[0]
            audio_sources.append(audio)

        stems = np.stack(audio_sources, axis=0)

        mixture = stems.sum(0)

        audio_path = f"{self.ozone_root}/ozone_train_fixed/{song_name}.wav"
        mixture_limited = librosa.load(audio_path, mono=False, sr=self.sample_rate)[0]

        mixture_lufs = self.dict_valid_loudness[song_name]
        augmented_gain = self.target_loudnorm_lufs - mixture_lufs
        mixture_loudnorm = mixture * db2linear(augmented_gain)

        return mixture_limited, mixture_loudnorm, song_name, mixture_lufs
