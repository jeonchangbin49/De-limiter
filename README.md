# De-limiter
An official repository of "Music De-Limiter Networks via Sample-wise Gain Inversion", which was presented at WASPAA 2023.

You can try the De-limiter with our Demo (https://huggingface.co/spaces/jeonchangbin49/De-limiter)

Audio Samples (https://catnip-leaf-c6a.notion.site/Music-De-limiter-7072c0e725fd42249ff78cbbaedc95d7?pvs=4)

Musdb-XL-train dataset (https://zenodo.org/records/12194067)

Paper (https://arxiv.org/abs/2308.01187)


## Note (June 21, 2024)
While working on my PhD thesis, I discovered some errors in the proposed dataset and mistakes in my training codes. Specifically, about 7% of the training data (ozone_seg_0.wav ~ ozone_seg_20000.wav) had slight phase shift problems, and the channel-swapping function in the training data loader was misused.

Currently, those are corrected. There are some changes on the experimental results in the paper. The model weights in this repo and the huggingface demo page are updated. If you are already using the musdb-XL-train dataset, sorry for the inconvenience, and please check the updated version. 

## Inference
All you need to do is just git clone this repository and install some of the requirements, and run the "inference.py" code.

```
git clone https://github.com/jeonchangbin49/De-limiter.git
```

You don't need to download separate model weight file, it is just contained in "./weight" directory. Thanks to the advantage of the SGI framework, it is only 9Mb. 

Put all of the music files (in .wav or .mp3 format) you want to de-limit in "./input_data" folder, or specify the "--data_root" argument, or just specify the path of the music file. Then, run the following code. 

```
# "--data_root=./input_data" or "--data_root=/path/to/music.wav", --output_directory=./output
python -m inference
```

De-limiter outputs will be saved in "./output" folder.

## Training
Check the detailed training configurations in "./configs/delimit_6_s.yaml"

```
python -m main_ddp --config=delimit_6_s
```

### Musdb-XL-train
We present the musdb-XL-train dataset for training de-limiter networks.

The musdb-XL-train dataset consists of a limiter-applied 300,000 segments of 4-sec audio segments and the 100 original songs. For each segment, we randomly chose arbitrary segment in 4 stems (vocals, bass, drums, other) of musdb-HQ training subset and randomly mixed them. Then, we applied a commercial limiter plug-in to each stem.

Once you finish the download, you have to unzip it. The data is about 200~210GB so please be sure to make enough space. Due to the copyright issue, the dataset contains the sample-wise gain parameters (in .npy files), instead of a wave file itself, to make each wave file of musdb-XL-train data from the musdb18-HQ dataset.

Please follow these steps to get the actual wave files of musdb-XL-train data.

```
# Move to the directory where you downloaded the musdb-XL-train data
# Then unzip the downloaded file
tar -xvf musdb-XL-train.tar.xz
# Back to our codes, move to "prepro" folder
# Run save_musdb_XL_train_wave.py
python save_musdb_XL_train_wave.py \
 --root=/path/to/musdb18hq \
 --musdb_XL_train_npy_root=/path/to/musdb-XL-train \
 --output=/path/to/musdb-XL-train
```

After finishing the data processing step, you can remove the "np_ratio" folder that contains the sample-wise gain ratio parameters but you should keep your csv files because they will be used in our training process.

Notice that our previous musdb-XL (https://zenodo.org/record/7041331) data is an evaluation dataset, and musdb-XL-train is a training dataset.

#### Dataset Construction

For a commercial limiter plug-in, we used the iZotope Ozone 9 Maximizer, following our previous work, musdb-XL, which is a mastering-finished (in terms of a limiter, not an EQ) version of musdb-HQ test subset.

The threshold parameters (related to the amount of a limiter operated) of the Ozone 9 Maximizer were chosen targeting the randomly selected loudness that sampled from the Gaussian distribution (mean -8, std 1). Parameters of the Gaussian distribution were selected following statistics of recent pop music (Refer the Table 1. of our previous paper, https://arxiv.org/abs/2208.14355).

The character parameters (related to the attack and release parameters) of the limiter were randomly sampled from the gamma distribution (a=2, scale=1, in https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html). 

The information on random mix parameters (gain and channel swap) is contained as csv files in our dataset.


