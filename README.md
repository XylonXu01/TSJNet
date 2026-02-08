# Towards Multi-Scenario Multi-modal Image Fusion: A UAV Benchmark and A Dual-task driven Target and Semantic Awareness Network

[![arXiv](https://img.shields.io/badge/arXiv-2402.01212-b31b1b.svg)](https://arxiv.org/abs/2402.01212)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-yellow.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-red.svg)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-UMS-orange.svg)](https://pan.baidu.com/s/1255aOfsLBvMm6ywlsyb15w?pwd=geph)

> **Yuchan Jie, Yushen Xu, Xiaosong Li, Haishu Tan**
>


This is the official PyTorch implementation of **TSJNet**, a multi-modality target and semantic awareness joint-driven image fusion network. TSJNet jointly leverages object detection and semantic segmentation information to guide the fusion of infrared and visible images.

<p align="center">
  <img src="fig/Framkwork.png" width="90%">
</p>

## 📖 Abstract

This study aims to address the problem of incomplete information in unimodal images for semantic segmentation and object detection tasks. Existing multimodal fusion methods suffer from limited capability in discriminative modeling of multi-scale semantic structures and salient target regions, which further restricts the effective fusion of task-related semantic details and target information across modalities. To tackle these challenges, this paper proposes a novel fusion network termed TSJNet, which leverages the semantic information output by high-level tasks in a joint manner to guide the fusion process. Specifically, we design a multi-dimensional feature extraction module with dual parallel branches to capture multi-scale and salient features. Meanwhile, a data-agnostic spatial attention module embedded in the decoder dynamically calibrates attention allocation across different data domains, significantly enhancing the model's generalization ability. To optimize both fusion and advanced visual tasks, we balance performance by combining fusion loss with semantic losses. Additionally, we have developed a multimodal unmanned aerial vehicle (UAV) dataset covering multiple scenarios (UMS). Extensive experiments demonstrate that TSJNet achieves outstanding performance on five public datasets (MSRS, M\textsuperscript{3}FD, RoadScene, LLVIP, and TNO) and our UMS dataset. The generated fusion results exhibit favorable visual effects, and compared to state-of-the-art methods, the mean average precision (mAP@0.5) and mean intersection over union (mIoU) for object detection and segmentation, respectively, improve by 7.97\% and 10.88\%.The code and the dataset has been publicly released at https://github.com/XylonXu01/TSJNet.

## ✨ Highlights

- 🔗 **Joint-driven Framework**: A serial architecture that jointly optimizes fusion, detection, and segmentation in an end-to-end manner.
- 🧩 **Local Significant Feature Extraction (LSFE)**: A dual-branch module combining a **Neighborhood Attention Transformer (NAT)** branch for global semantic features and a **Channel Operation with Attention (COA)** branch for local detail features.
- 🔄 **ResNeSt Backbone with Split Attention**: Utilizes ResNeSt blocks with split attention mechanism for robust multi-scale feature extraction in both encoder and decoder.
- 📊 **Multi-task Loss**: Combines fusion loss (gradient, intensity, SSIM), detection loss, and segmentation loss with attention-based diversity regularization for comprehensive optimization.
- 🏆 **State-of-the-art Performance**: Achieves superior fusion quality while simultaneously improving downstream detection (+7.97% mAP@0.5) and segmentation (+10.88% mIoU) performance.
- 🛩️ **UAV Multi-Scenario Dataset (UMS)**: A newly constructed multimodal UAV benchmark covering 6 diverse real-world scenarios with paired infrared-visible images and dense annotations.

## 🌐 UMS Dataset

We construct a new **UAV Multi-Scenario (UMS)** multimodal dataset to address the scarcity of UAV-perspective infrared-visible image fusion benchmarks. The dataset is captured by a **DJI M30T** drone equipped with both visible and infrared cameras, covering **6 diverse real-world scenarios**.

<p align="center">
  <img src="fig/UMS.png" width="95%">
</p>

### 📋 Scenario Overview

| Scenario | Description |
|----------|-------------|
| 🅿️ **Parking Lot** | Vehicles and pedestrians in parking areas |
| 🏫 **Campus** | Campus roads with pedestrians, bicycles, and vehicles |
| 🛣️ **Motorways** | Highway scenes with dense traffic flow |
| 🏙️ **Street Scenes** | Urban street views with mixed traffic and pedestrians |
| 🏘️ **Residential** | Residential area with buildings and sparse targets |
| 🌳 **Park** | Park environment with complex structures and vegetation |

### 📊 Dataset Features

- **Modalities**: Paired visible (RGB) and infrared (thermal) images captured simultaneously
- **Platform**: DJI Matrice 30T UAV with dual-sensor gimbal
- **Perspective**: Aerial / bird's-eye view
- **Annotations**: Object detection bounding boxes + semantic segmentation masks
- **Scenarios**: 6 representative real-world scenes (Parking Lot, Campus, Motorways, Street Scenes, Residential, Park)
- **Categories**: Person, Car, Bicycle, Motorcycle, and more

### ⬇️ Download

| Source | Link |
|--------|------|
| Baidu Netdisk | [Download](https://pan.baidu.com/s/1255aOfsLBvMm6ywlsyb15w?pwd=geph) (`geph`) |

After downloading, organize the dataset as follows:
```
data/
└── UMS/
    ├── ir/            # Infrared images
    ├── vi/            # Visible images

```

## 🔥 Visual Results

### Qualitative Comparison

<p align="center">
  <img src="fig/radar_map.png" width="90%">
</p>

### Fusion Results on Multiple Datasets

<p align="center">
  <img src="fig/subjective_assessment.png" width="90%">
</p>



## ⚙️ Environment Setup

### Requirements

- Python >= 3.8
- PyTorch >= 1.12.0
- CUDA >= 11.3
- [NATTEN](https://github.com/SHI-Labs/NATTEN) (Neighborhood Attention)

### Installation

```bash
# Clone the repository
git clone https://github.com/XylonXu01/TSJNet.git
cd TSJNet

# Create conda environment (recommended)
conda create -n tsjnet python=3.10
conda activate tsjnet

# Install PyTorch (adjust CUDA version as needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install NATTEN (Neighborhood Attention, must match PyTorch & CUDA versions)
# See https://github.com/SHI-Labs/NATTEN for installation guide
pip install natten

# Install other dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | >= 1.12.0 | Deep learning framework |
| `torchvision` | >= 0.13.0 | Pretrained models (Faster R-CNN, DeepLabV3) |
| `natten` | latest | Neighborhood Attention (NAT) |
| `timm` | latest | DropPath, trunc_normal_ utilities |
| `einops` | latest | Tensor rearrangement |
| `opencv-python` | latest | Image I/O |
| `wandb` | latest | Training logging & visualization |
| `tqdm` | latest | Progress bars |

## 🚀 Usage

### Inference

Download the pretrained weights and place them in the `weight/` directory.

Prepare your test images in the following structure:
```
test_img/
└── MSRS/          # or M3FD, RoadScene, LLVIP
    ├── ir/        # Infrared images
    └── vi/        # Visible images
```

Run inference:

```bash
# Default: use GPU 0, MSRS dataset
python inference.py

# Specify GPU and dataset
python inference.py --gpu 0 --dataset MSRS

# Full options
python inference.py \
    --gpu 0 \
    --ckpt ./weight/TSJNet.pth \
    --dataset MSRS \
    --test_folder ./test_img \
    --save_folder ./test_result

# Convert grayscale fusion results to colorized images (YCbCr color space transformation)
python Funsion_image_2_RGB.py
```

#### Inference Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--gpu` | `0` | GPU device id |
| `--ckpt` | `./weight/TSJNet.pth` | Path to pretrained checkpoint |
| `--dataset` | `MSRS` | Dataset name (`MSRS`, `M3FD`, `RoadScene`, `LLVIP`) |
| `--test_folder` | `./test_img` | Root folder of test images |
| `--save_folder` | `./test_result` | Output folder for fused results |

### Training

Training jointly optimizes the fusion, detection, and segmentation subnetworks:

```bash
python TSJNet_train.py
```

#### Training Configuration

Key hyperparameters are configured in `TSJNet_train.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_epochs` | 100 | Total training epochs |
| `lr` | 0.0001 | Initial learning rate |
| `weight_decay` | 0.01 | L2 regularization |
| `batch_size` | 2 | Batch size |
| `coeff_mse_loss_VF` | 6 | VIS reconstruction loss weight (α₁) |
| `coeff_decomp` | 2.0 | Decomposition loss weight (α₂, α₄) |
| `att_weight` | 0.1 | Attention diversity loss weight |
| `num_classes` | 6 | Number of segmentation classes (MSRS) |

#### Dataset Preparation

Training requires the **MSRS** dataset with:
- Infrared and visible image pairs
- VOC-format object detection annotations (bounding boxes + labels)
- Semantic segmentation masks

Configure dataset paths in `TSJNet_train.py`:
```python
ir_VOC_root = './data/MSRS'
vi_VOC_root = './data/MSRS'
```



#### Training Monitoring

Training metrics are logged to [Weights & Biases (wandb)](https://wandb.ai/):
```
fusion_loss, fusion_seg_loss, fusion_det_losses, ir_det_losses, vi_det_losses, total_loss, avg_loss
```

## 📊 Results

### Object Detection (mAP@0.5)

<p align="center">
  <img src="fig/detection.png" width="80%">
</p>

### Semantic Segmentation (mIoU)

<p align="center">
  <img src="fig/segmentation.png" width="80%">
</p>



## 📄 Citation

If you find this work useful, please cite:

```bibtex
@article{TSJNet,
  title={TSJNet: A multi-modality target and semantic awareness joint-driven image fusion network},
  author={Jie, Yuchan and Xu, Yushen and Li, Xiaosong and Tan, Haishu},
  journal={arXiv preprint arXiv:2402.01212},
  year={2024}
}
```

## 📬 Contact

If you have any questions, feel free to open an issue or contact us.

## 🙏 Acknowledgement

This work is built upon several excellent open-source projects:
- [NATTEN](https://github.com/SHI-Labs/NATTEN) - Neighborhood Attention Transformer
- [timm](https://github.com/huggingface/pytorch-image-models) - PyTorch Image Models
- [CDDFuse](https://github.com/Zhaozixiang1228/MMIF-CDDFuse) - CDDFuse

## License

This project is released under the [MIT License](LICENSE).
