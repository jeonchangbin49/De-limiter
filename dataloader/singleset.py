import math

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

# Modified version from woosungchoi's original implementation
class SingleTrackSet(Dataset):
    def __init__(self, track, hop_length, num_frame=128, target_name="vocals"):

        assert len(track.shape) == 2
        assert track.shape[0] == 2  # check stereo audio

        self.hop_length = hop_length
        self.window_length = hop_length * (num_frame - 1)  # 130048
        self.trim_length = self.get_trim_length(self.hop_length)  # 5120

        self.true_samples = self.window_length - 2 * self.trim_length  # 119808

        self.lengths = [track.shape[1]]  # track lengths (in sample level)
        self.source_names = [
            "vocals",
            "drums",
            "bass",
            "other",
        ]  # == self.musdb_train.targets_names[:-2]

        self.target_names = [target_name]

        self.num_tracks = 1

        import math

        num_chunks = [
            math.ceil(length / self.true_samples) for length in self.lengths
        ]  # example : 44.1khz 180sec audio, => [67]
        self.acc_chunk_final_ids = [
            sum(num_chunks[: i + 1]) for i in range(self.num_tracks)
        ]  # [67]

        self.cache_mode = True
        self.cache = {}
        self.cache[0] = {}
        self.cache[0]["linear_mixture"] = track

    def __len__(self):
        return self.acc_chunk_final_ids[-1] * len(self.target_names)  # 67

    def __getitem__(self, idx):

        target_offset = idx % len(self.target_names)  # 0
        idx = idx // len(self.target_names)  # idx

        target_name = self.target_names[target_offset]  # 'vocals'
        mixture_idx, start_pos = self.idx_to_track_offset(
            idx
        )  # idx * self.true_samples

        length = self.true_samples
        left_padding_num = right_padding_num = self.trim_length  # 5120
        if mixture_idx is None:
            raise StopIteration
        mixture_length = self.lengths[mixture_idx]
        if start_pos + length > mixture_length:  # last
            right_padding_num += self.true_samples - (mixture_length - start_pos)
            length = None

        mixture = self.get_audio(mixture_idx, "linear_mixture", start_pos, length)
        mixture = F.pad(mixture, (left_padding_num, right_padding_num), "constant", 0)

        return mixture

    def idx_to_track_offset(self, idx):

        for i, last_chunk in enumerate(self.acc_chunk_final_ids):
            if idx < last_chunk:
                if i != 0:
                    offset = (idx - self.acc_chunk_final_ids[i - 1]) * self.true_samples
                else:
                    offset = idx * self.true_samples
                return i, offset

        return None, None

    def get_audio(self, idx, target_name, pos=0, length=None):
        track = self.cache[idx][target_name]
        return track[:, pos : pos + length] if length is not None else track[:, pos:]

    def get_trim_length(self, hop_length, min_trim=5000):
        trim_per_hop = math.ceil(min_trim / hop_length)

        trim_length = trim_per_hop * hop_length
        assert trim_per_hop > 1
        return trim_length

