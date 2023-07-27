# De-limiter
An official repository of "Music De-limiter Networks via Sample-wise Gain Inversion", which will be presented in WASPAA 2023.

Still, under construction... paper, demo, data will be added soon.


## Inference
All you need to do is just git clone this repository and install some of the requirements, and run the "inference.py" code.

```
git clone https://github.com/jeonchangbin49/De-limiter.git
```

You don't need to download separate model weight file, it is just contained in "./weight" directory. Thanks to the advantage of the SGI framework, it is only 9Mb. 

Put all of the music files (in .wav of .mp3 format) you want to de-limit in "./input_data" folder or specify the "--data_root" argument. Then, run the following code. 

```
# --data_root=./input_data --output_directory=./output
python -m inference
```

De-limiter outputs will be saved in "./output" folder.

## Training
Check the detailed training configurations in "./configs/delimit_6_s.yaml"

```
CUDA_VISIBLE_DEVICES=0 python -m main_ddp --config=delimit_6_s
```

### Musdb-XL-train
We have made musdb-XL-train dataset for training de-limiter networks.

Detailed explanations and data itself will be added soon.