import os
import json

from torch.utils.data import DataLoader
import soundfile as sf
import tqdm

from dataloader import DelimitValidDataset


def main():
    # Parameters
    data_path = "/data1/Music/musdb18hq"
    # save_path = "/data2/personal/jeon/delimit/data/musdb18hq_limited_custom_limiter"
    save_path = (
        "/data2/personal/jeon/delimit/data/musdb18hq_custom_limiter_fixed_attack"
    )
    batch_size = 1
    num_workers = 1
    sr = 44100

    # Dataset
    # dataset = DelimitValidDataset(root=data_path, use_custom_limiter=True, custom_limiter_attack_range=[2.0,2.0])
    # With fixed attack
    dataset = DelimitValidDataset(
        root=data_path, use_custom_limiter=True, custom_limiter_attack_range=[2.0, 2.0]
    )
    data_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )
    dict_valid_loudness = {}
    dict_limiter_params = {}
    # Preprocessing
    for (
        limited_audio,
        orig_audio,
        audio_name,
        loudness,
        custom_attack,
        custom_release,
    ) in tqdm.tqdm(data_loader):
        audio_name = audio_name[0]
        limited_audio = limited_audio[0].numpy()
        loudness = float(loudness[0].numpy())
        dict_valid_loudness[audio_name] = loudness
        dict_limiter_params[audio_name] = {
            "attack_ms": float(custom_attack[0].numpy()),
            "release_ms": float(custom_release[0].numpy()),
        }
        # Save audio
        os.makedirs(os.path.join(save_path, "valid"), exist_ok=True)
        audio_path = os.path.join(save_path, "valid", audio_name)
        sf.write(f"{audio_path}.wav", limited_audio.T, sr)
        # write json write code
    with open(os.path.join(save_path, "valid_loudness.json"), "w") as f:
        json.dump(dict_valid_loudness, f, indent=4)
    with open(os.path.join(save_path, "valid_limiter_params.json"), "w") as f:
        json.dump(dict_limiter_params, f, indent=4)


if __name__ == "__main__":
    main()
