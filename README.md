# Multi-Bridge Denoising Diffusion Probabilistic Models (NeurIPS 2026)

![result-gif1](./figures/imagenet.gif)

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/NeuroSML/mb-ddpm.git
cd mb-ddpm
```

### 2. Set up the environment

We recommend using **Conda** to create an isolated Python environment.

```bash
conda create -n mb-ddpm python=3.9
conda activate mb-ddpm
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

Then install the CUDA 12.1 version of PyTorch:

```bash
pip install torch==2.1.0+cu121 \
            torchvision==0.16.0+cu121 \
            torchaudio==2.1.0+cu121 \
            --extra-index-url https://download.pytorch.org/whl/cu121
```

> **Note:** Make sure your NVIDIA driver and CUDA environment are compatible with CUDA 12.1.

## Reproduce the Imagenet experiment

### 1. Download the pretrained checkpoint

Download the pretrained checkpoint `model_params_conditional.pth` and `model_params_marginal.pth` from the [link](https://drive.google.com/drive/folders/1VbaoYFCjt0_9XdLp7Is-QW49Xss4RBAM?usp=sharing) and place it in the `models/` directory.

```bash
mkdir models
mv {DOWNLOAD_DIR}/model_params_conditional.pth ./models/
mv {DOWNLOAD_DIR}/model_params_marginal.pth ./models/
```

Replace `{DOWNLOAD_DIR}` with the path to the directory where you downloaded the checkpoint.

Your directory should look like:

```text
mb-ddpm/
├── models/
│   └── model_params_conditional.pth
│   └── model_params_marginal.pth
└── ...
```

### 2. Sampling

We provide two samples from the ImageNet test set in the `./samples` directory.

To generate samples using the pretrained models, run:

```bash
python3 sample.py \
    --data_config=configs/imagenet_config.yaml \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/diffusion_config.yaml
```

### 3. Training

You can train the models from scratch instead of downloading the pretrained checkpoints. 

Run the following command:

```bash
python3 train.py \
    --data_config=configs/imagenet_config.yaml \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/diffusion_config.yaml \
    --training_config=configs/training_config.yaml \
    --method={METHOD}
```
For multi-process training replace `python3` with `torchrun --nproc_per_node=$NUM_GPUS`.

The `METHOD` argument must be set to one of the following:

* `conditional`
* `marginal`

Run the command twice, once for each method, to train both conditional and marginal models. 



  
## Train and Sample on Different Data


### 1. Data Directory Structure

The data directory should contain one subdirectory for each source image and one for the target image. For the ImageNet experiment, the sources are `jpeg5`, `blurgauss`, `noisegauss`, and `sr4xbicubic`, while the target is `clean`.

The expected directory structure is:

```text
samples/
├── jpeg5/
├── blurgaus/
├── noisegauss/
├── sr4xbicubic/
└── clean/
```

Each subdirectory should contain the same number of images. Corresponding images across modalities must share the same `image_id`.

Images should follow the naming convention:

```text
{image_id}_{modality}.png
```

For example:

```text
samples/
├── clean/
│   ├── 0001_clean.png
│   ├── 0002_clean.png
│   └── ...
├── blurgauss/
│   ├── 0001_blurgauss.png
│   ├── 0002_blurgauss.png
│   └── ...
├── noisegauss/
│   ├── 0001_noisegauss.png
│   ├── 0002_noisegauss.png
│   └── ...
└── ...
```



### 2. Training

To train models on your own data sources, first create a data configuration file in the `configs/` directory. The `{YOUR_DATA}_config.yaml` file specifies the data directory, source modalities, and target modality. For example:

```yaml
data_dir: ./samples/
source_name: blurgaus,jpeg5,sr4xbicubic,noisegauss
target_name: clean
```

Ensure that your data directory follows the structure and file-naming conventions described in the [Data Directory Structure](#data-directory-structure) section.

You can also modify the model, diffusion, and training configuration files in `configs/` to adjust the corresponding parameters.

Then run the following command:

```bash
python3 train.py \
    --data_config=configs/{YOUR_DATA}_config.yaml \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/diffusion_config.yaml \
    --training_config=configs/training_config.yaml \
    --method={METHOD}
```
For multi-process training replace `python3` with `torchrun --nproc_per_node=$NUM_GPUS`.

The `METHOD` argument must be one of the following:

* `conditional`
* `marginal`

Both methods must be run to train the conditional and marginal models.


### 3. Sampling

To generate samples using the trained models, run:

```bash
python3 sample.py \
    --data_config=configs/{YOUR_DATA}_config.yaml \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/diffusion_config.yaml
```

Make sure the appropriate model checkpoint is available and that the configuration settings match the model you wish to sample from.


## Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{Monod2026,
  series = {NeurIPS 2026},
  title = {Multi-Bridge Denoising Diffusion Probabilistic Models},
  booktitle = {40th Conference on Neural Information Processing Systems (NeurIPS 2026)},
  author = {Monod, Mélodie and Ganjgahi, Habib},
  year = {2026},
}
```


