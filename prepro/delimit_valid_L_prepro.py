import os
import json

from torch.utils.data import DataLoader
import soundfile as sf
import tqdm

from dataloader import DelimitValidDataset


def main():
    # Parameters
    data_path = "/path/to/musdb18hq"
    save_path = "/path/to/musdb18hq_limited_L"
    batch_size = 1
    num_workers = 1
    sr = 44100

    # Dataset
    dataset = DelimitValidDataset(root=data_path, valid_target_lufs=-14.39)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )
    dict_valid_loudness = {}
    # Preprocessing
    for limited_audio, orig_audio, audio_name, loudness in tqdm.tqdm(data_loader):
        audio_name = audio_name[0]
        limited_audio = limited_audio[0].numpy()
        loudness = float(loudness[0].numpy())
        dict_valid_loudness[audio_name] = loudness
        # Save audio
        os.makedirs(os.path.join(save_path, "valid"), exist_ok=True)
        audio_path = os.path.join(save_path, "valid", audio_name)
        sf.write(f"{audio_path}.wav", limited_audio.T, sr)
        # write json write code
    with open(os.path.join(save_path, "valid_loudness.json"), "w") as f:
        json.dump(dict_valid_loudness, f, indent=4)


if __name__ == "__main__":
    main()
