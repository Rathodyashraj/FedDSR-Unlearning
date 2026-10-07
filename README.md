<div align="center">

#  FedDSR: Codebook-Based Distribution Alignment for Heterogeneous Federated CT Super-Resolution

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Federated Learning](https://img.shields.io/badge/Federated%20Learning-8A2BE2.svg?style=for-the-badge)](#)
[![Medical Imaging](https://img.shields.io/badge/Medical%20Imaging-00a2e8.svg?style=for-the-badge)](#)
[![Super Resolution](https://img.shields.io/badge/Super%20Resolution-ff69b4.svg?style=for-the-badge)](#)
[![Architecture](https://img.shields.io/badge/Architecture-VQ--VAE--2-f39c12.svg?style=for-the-badge)](#)
[![Data](https://img.shields.io/badge/Data-Multi--Organ%20CT-4caf50.svg?style=for-the-badge)](#)
[![Setting](https://img.shields.io/badge/Setting-Non--IID-e74c3c.svg?style=for-the-badge)](#)

</div>

---

## 📖 Abstract / Overview

Medical Image Super-Resolution (SR) is critical for downstream diagnostic tasks, but acquiring large-scale, centralized, high-resolution (HR) medical data is often prohibited by strict privacy regulations. Federated Learning (FL) mitigates privacy concerns by enabling decentralized training; however, it introduces severe challenges, specifically modality heterogeneity and excessive communication overhead.

<img width="4068" height="2913" alt="combined" src="https://github.com/user-attachments/assets/7a17010e-f1c4-4681-98ba-e962d641c1f8" />

In this repository, we present the official PyTorch implementation for our double-blind submission. We propose **Distribution-Aware Federated Medical Super-Resolution**, a novel framework leveraging a hierarchical **VQ-VAE-2** architecture combined with a semantic-aware federated aggregation strategy. Our approach explicitly tackles extreme non-IID settings across multi-organ CT datasets by utilizing latent codebook vitality and semantic centrality to dynamically compute optimal aggregation weights. This effectively achieves substantial qualitative improvements over baseline aggregation strategies while simultaneously minimizing the required communication bandwidth.

## 🚀 Motivation

- **Data Privacy**: Medical data cannot be easily centralized. FL enables collaborative training without sharing raw data.
- **Heterogeneous Medical Modalities**: Multi-institutional data often exhibits high variance in contrast, noise, and organ distributions. Standard aggregation (e.g., FedAvg) struggles under these non-IID conditions.
- **Computation Overhead**: Transmitting dense super-resolution network parameters or high-dimensional features is costly. Our architecture relies on a highly efficient discrete latent space.

## ✨ Key Contributions

1. **Hierarchical VQ-VAE-2 for SR**: We design a multi-scale encoder-decoder architecture with efficient cross-level attention, projecting high-dimensional medical images into a compact, discrete latent space for highly efficient transmission.
2. **Distribution-Aware Aggregation**: A novel federated aggregation algorithm that leverages semantic centrality and codebook vitality metrics to dynamically weight client updates, intrinsically mitigating the adverse effects of non-IID data distributions.
3. **Comprehensive Multi-Organ Benchmarking**: We validate our methods on a severely non-IID setup spanning four distinct clinical CT datasets (COVID-19, Pancreas, Kidney, Brain Stroke), demonstrating superior edge-preservation and artifact reduction.

---

## 🧠 Methodology & Architecture

Our architecture features a robust top-and-bottom codebook quantization scheme combined with a conditional hierarchical decoder. The global learning objective meticulously balances local reconstruction fidelity with perceptual quality by minimizing a hybrid loss formulation:

- **$\mathcal{L}_{\text{Recon}}$ (Charbonnier Loss)**: Ensures robust pixel-level structural reconstruction while mitigating sensitivity to outliers.
- **$\mathcal{L}_{\text{LPIPS}}$ (Perceptual Loss)**: Preserves high-level deep-feature representations extracted via VGG networks.
- **$\mathcal{L}_{\text{SSIM}}$ (Multi-Scale Structural Similarity)**: Accurately reconstructs the diagnostic structural integrity of the specific organ.
- **$\mathcal{L}_{\text{Edge}}$ (Edge-Aware Loss)**: Enhances high-frequency texture boundaries crucially required by radiologists.

### System Pipeline

## 📊 Experimental Setup

### Datasets

The framework is evaluated under a decentralized, non-IID scenario utilizing four distinct public CT datasets corresponding to different organs/pathologies, simulated as 4 separate clients:

1. **Client 0**: COVID-19 CT
2. **Client 1**: Pancreas CT
3. **Client 2**: Kidney CT (Normal, Cyst, Tumor, Stone)
4. **Client 3**: Brain Stroke CT

### Evaluation Metrics

We systematically evaluate both the distortion and perceptual quality of the reconstructed images using standard quantitative metrics:

- **PSNR** (Peak Signal-to-Noise Ratio): Evaluates pixel-level reconstruction accuracy.
- **SSIM** (Structural Similarity Index): Measures perceptual structural differences.
- **LPIPS** (Learned Perceptual Image Patch Similarity): Assesses deep semantic feature alignment.
- **FID** (Fréchet Inception Distance): Evaluates the overall distribution distance between generated and ground-truth super-resolutions.

---

## 📈 Results & Analysis

Our proposed distribution-aware federated aggregation algorithm heavily surpasses naive decentralized methods (e.g., FedAvg and FedProx) across all benchmarked clinical datasets. The federated global model converges stably to high-fidelity reconstructions, demonstrating robust cross-organ feature preservation.

Specifically, our approach yields:

- **Statistically Significant Improvements in PSNR and SSIM**: Accurately producing high-frequency textures that directly aid diagnosis.
- **Superior LPIPS and FID Scores**: Confirming that the generated super-resolutions closely align with real, high-resolution latent representations without hallucination artifacts.
- **Robustness against Non-IID Modality Shifts**: Effectively preventing catastrophic forgetting and bias toward clients possessing larger dataset sizes.

*(Note: Exact quantitative tables will be updated upon paper acceptance. Please refer to the manuscript for comprehensive benchmarking.)*


## 🛠️ Installation

**1. Clone the repository**

```bash
git clone https://github.com/xzarulin7/FedDSR.git
cd FedDSR
```

**2. Create a virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

*(Dependencies include `torch`, `torchvision`, `lpips`, `piq`, `torchmetrics`, `numpy`, The requirements file is inferred from imports).*

---

## 💻 Usage

### ⚙️ Configuration

All hyperparameters, paths, and training settings are centralized in `config.py`. Ensure dataset paths are correctly mapped before running.

### 🏃‍♂️ Running the Pipeline

**Stage 1: Federated Training (Global Model Aggregation)**
Train the VQ-VAE-2 distributed across clients using our proposed distribution-aware aggregation strategy.

```bash
python main.py --rounds 30 --clients 4 --patience 15 --gpu 0 --aggregator FedMedSR
```

**Stage 2: Client Personalization (Decoder Fine-Tuning)**
Following federated aggregation, fine-tune the decoder locally for optimized organ-specific results.

```bash
python stage2_decoder_finetune.py --epochs 10 --clients 4 --lr 5e-6 --gpu 0
```

**Stage 3: Evaluation & FID Calculation**
Calculate cumulative FID scores and produce visual comparisons.

```bash
python calculate_fid.py --clients 4 --gpu 0
```

---

## 📂 Repository Structure

```text
.
├── config.py                  # Global hyperparameters & settings
├── data.py                    # Dataset definitions & dataloaders
├── models.py                  # Hierarchical VQ-VAE-2 architectures
├── client.py                  # Local client training & evaluation logic
├── server.py                  # Distribution-aware federated aggregation logic
├── main.py                    # Entry point for Stage 1 Federated Learning
├── stage2_decoder_finetune.py # Entry point for local personalization
├── calculate_fid.py           # Evaluation script for generated images
├── utils.py                   # Custom composite loss functions & loggers
└── README.md                  # This documentation
```











