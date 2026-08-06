# TS-FDNet

This repository provides the code for the method in our paper '**Text-Semantics Guided Frequency-Decoupled Network for Semantic Change Detection**'. 

## Overview

<p align="center">
  <img src="image/Overview.png" width="800">
</p>

**TS-FDNet** is a multimodal, frequency-decoupled deep learning network tailored for remote sensing semantic change detection (SCD). By separating feature frequencies and injecting text-semantic priors, the network successfully suppresses global pseudo-changes and resolves category confusion under complex environmental variations.Specifically, bi-temporal images are first encoded into multi-scale visual features, which are enhanced at high decoder layers. The WFD module then decouples low-frequency global structures and high-frequency local details, which are modeled by Mamba and deformable convolutions, respectively. Finally, bidirectional change-semantic feedback jointly refines change localization and bi-temporal semantic predictions.



## Environment setup

TS-FDNet requires a CUDA-enabled GPU. Create a Python environment, install a PyTorch/torchvision build compatible with your CUDA version, and then install the remaining dependencies:

```bash
git clone https://github.com/SDUSTVIGroup/TSFDNet.git
cd TSFDNet

conda create -n tsfdnet python=3.10 -y
conda activate tsfdnet

# Install PyTorch and torchvision according to your CUDA version:
# https://pytorch.org/get-started/locally/

pip install numpy scipy scikit-image matplotlib opencv-python tensorboard tensorboardX timm
pip install causal-conv1d mamba-ssm
pip install git+https://github.com/openai/CLIP.git
```


## Data preparation

Taking training the SECOND dataset as an example, split the SCD data into training, validation and testing (if available) set and organize them as follows:

>SECOND
>  - train
>    - im1
>    - im2
>    - label1
>    - label2
>  - val
>    - im1
>    - im2
>    - label1
>    - label2
>  - test
>    - im1
>    - im2
>    - label1
>    - label2
    
If the downloaded SECOND annotations are RGB color masks, either convert them to single-channel index masks before training or enable `Color2Index` in `datasets/RS_SECOND.py` for both `label1` and `label2`.

Set the dataset root in `datasets/RS_SECOND.py`:

```python
root = 'your dataset'
```

The pretrained weights can be accessed at “Link: [Baidu Netdisk](https://pan.baidu.com/s/1t-Gu3oO1pqJggPxjVbSPfA?pwd=gypx) (提取码: `gypx`)


## Train
```bash
python train_TSFDNet.py 
```

## Checkpoint

- **TS-FDNet checkpoint**：[Baidu Netdisk](https://pan.baidu.com/s/1sW-WgOfNvPb3HVh4LSOtwA?pwd=rprp) (提取码: `rprp`)

## Results

<p align="center">
  <img src="image/Quantitative Comparison Results on the SECOND Dataset.png" width="800">
</p>


<p align="center">
  <img src="image/Quantitative Comparison Results on the JL1 and Landsat-SCD Datasets.png" width="800">
</p>


## Acknowledgement
