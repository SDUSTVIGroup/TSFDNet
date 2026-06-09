# TS-FDNet

This repository provides the code for the method in our paper '**Text-Semantics Guided Frequency-Decoupled Network for Semantic Change Detection**'. 

## Overview

<p align="center">
  <img src="Overview.png" width="800">
</p>

**TS-FDNet** is a multimodal, frequency-decoupled deep learning network tailored for remote sensing semantic change detection (SCD). By separating feature frequencies and injecting text-semantic priors, the network successfully suppresses global pseudo-changes and resolves category confusion under complex environmental variations.

The framework primarily advances SCD through three core phases and modules:

- **CLIP-Guided Text Feature Injection Module**: Introduces category-level text semantic priors (dynamically optimized via Context Optimization, CoOp) into high-level bi-temporal visual features. It utilizes spatial- and channel-wise dual-path attention to enhance semantic discriminability and reduce visual ambiguity.
- **Wavelet Frequency Decomposition (WFD) Module**: Decouples bi-temporal features into high- and low-frequency components via 2D Discrete Wavelet Transform (DWT). The **low-frequency branch** captures global structural context and long-range dependencies using four-directional Mamba scanning, while the **high-frequency branch** employs Deformable Convolutions (DCN) to adapt to irregular local geometric variations.
- **Bidirectional Synergistic Decoding (SGSE & CGSA)**: Establishes a bidirectional semantic-change refinement mechanism. It contains the **Semantic-Guided Spatial Enhancement (SGSE)** module and the **Change-Guided Semantic Attention (CGSA)** module, enabling mutual optimization and joint enhancement of change localization and semantic recognition.

## Data preparation

Split the SCD data into training, validation and testing (if available) set and organize them as follows:

>YOUR_DATA_DIR
>  - Train
>    - im1
>    - im2
>    - label1
>    - label2
>  - Val
>    - im1
>    - im2
>    - label1
>    - label2
>  - Test
>    - im1
>    - im2
>    - label1
>    - label2
    
The pretrained weights can be accessed at “Link:

## Train
```bash
python train_TSFDNet.py 
```

## Checkpoint

- **TS-FDNet checkpoint**：

## Results

<p align="center">
  <img src="TSFDNet.png" width="800">
</p>

- **Visual Comparison on the SECOND Dataset**

<p align="center">
  <img src="TSFDNet.png" width="800">
</p>

- **Visual Comparison on the JL1 Dataset**

<p align="center">
  <img src="TSFDNet.png" width="800">
</p>

- **Results of the ablation study**

<p align="center">
  <img src="TSFDNet.png" width="800">
</p>

## Acknowledgement
