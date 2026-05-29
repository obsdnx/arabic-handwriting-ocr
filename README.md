# Lightweight Attention-Based CNNs for Arabic Handwriting Recognition

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)
[![Dataset: HIJJA](https://img.shields.io/badge/Dataset-HIJJA-green.svg)](https://github.com/m-aljhani/HIJJA)
[![Accuracy](https://img.shields.io/badge/Top--1%20Accuracy-93.2%25-brightgreen.svg)]()

---

## Overview

Arabic Handwriting Recognition (AHR) is a high-impact, underserved problem in computer vision. Arabic script's cursive structure, rich ligature system, and extreme writer variance make it fundamentally harder than Latin OCR — yet demand is growing fast across education technology, document digitization, and accessibility tooling.

This project delivers a **production-oriented, resource-efficient AHR system** trained on the [HIJJA dataset](https://github.com/m-aljhani/HIJJA) of isolated Arabic characters written by children. The final model achieves **93.2% top-1 accuracy** using a custom **ResNet-13 + Squeeze-and-Excitation (SE)** architecture designed for edge deployment — no GPU required at inference time.

---

## Results at a Glance

| Model                    | Top-1 Accuracy | Parameters | Notes                          |
|--------------------------|:--------------:|:----------:|--------------------------------|
| Baseline CNN             | 90.2%          | ~500K      | Tuned hyperparameters          |
| Baseline + CBAM          | 90.3%          | ~510K      | No significant gain            |
| Baseline + SE            | 90.7%          | ~505K      | Lightweight channel attention  |
| ResNet-13                | 92.8%          | ~620K      | Residual backbone              |
| **ResNet-13 + SE**       | **93.2%**      | **~630K**  | **Best model (this repo)**     |

All models trained on the HIJJA dataset (28 isolated Arabic character classes, child handwriting).

---

## Why This Architecture

Three deliberate design decisions drive the performance-efficiency tradeoff:

**1. Compact residual backbone (ResNet-13)**  
Standard ResNet variants are over-parameterized for small-alphabet character classification. ResNet-13 retains skip connections for gradient stability across fine-grained Arabic stroke features while cutting depth to what the task actually requires.

**2. Squeeze-and-Excitation blocks**  
SE blocks recalibrate channel-wise feature responses with less than 1% parameter overhead. For Arabic script — where stroke-level channel activations carry disproportionate signal — this targeted attention consistently outperforms both no-attention and spatial attention (CBAM) baselines.

**3. No attention vs. wrong attention**  
CBAM added spatial attention on top of channel attention. Ablation showed it hurt generalization on this dataset, likely due to spatial feature maps being too small post-pooling to benefit from spatial reweighting. Knowing when *not* to add a module is as important as knowing when to add one.

---

## Architecture

```
Input (64×64 grayscale character image)
        │
        ▼
[Conv2D 3×3 → BatchNorm → ReLU]       # Initial feature extraction
        │
        ▼
┌───────────────────────────────────────────────┐
│  ResNet-13 Block (×3, increasing channels)    │
│                                               │
│   ┌─ Conv2D 3×3 → BN → ReLU ─┐              │
│   │  Conv2D 3×3 → BN         │              │
│   │                           │              │
│   │  [SE Block]               │              │
│   │   Global AvgPool          │              │
│   │   FC (C → C/r) → ReLU    │              │
│   │   FC (C/r → C) → Sigmoid │              │
│   │   Channel-wise scale      │              │
│   │                           │              │
│   └──────── ⊕ skip ──────────┘              │
└───────────────────────────────────────────────┘
        │
        ▼
[Global Average Pooling]
        │
        ▼
[Linear → Softmax]  →  28 Arabic character classes
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/arabic-handwriting-ocr.git
cd arabic-handwriting-ocr

# Create a virtual environment
python -m venv venv && source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Requirements:** Python 3.8+, PyTorch 1.12+, torchvision, numpy, matplotlib, scikit-learn

---

## Dataset

This project uses the **HIJJA dataset** — a large-scale collection of isolated Arabic handwritten characters produced by school-age children, covering all 28 Arabic alphabet characters.

Download and structure the dataset as follows:

```
data/
└── hijja/
    ├── train/
    │   ├── alef/
    │   ├── ba/
    │   └── ...
    └── test/
        ├── alef/
        ├── ba/
        └── ...
```

---

## Usage

**Train the best model (ResNet-13 + SE):**

```bash
python train.py --model resnet13_se --epochs 50 --batch-size 64 --lr 1e-3
```

**Evaluate on the test set:**

```bash
python evaluate.py --model resnet13_se --checkpoint checkpoints/best.pth
```

**Run inference on a single image:**

```bash
python predict.py --image path/to/character.png --checkpoint checkpoints/best.pth
```

**Run all ablation experiments:**

```bash
python run_ablations.py  # Reproduces the results table above
```

---

## Project Structure

```
.
├── models/
│   ├── baseline_cnn.py       # Tuned baseline CNN
│   ├── attention.py          # SE and CBAM block implementations
│   └── resnet13.py           # ResNet-13 with optional SE integration
├── data/
│   └── dataset.py            # HIJJA dataloader and augmentation pipeline
├── train.py                  # Training loop with LR scheduling and checkpointing
├── evaluate.py               # Evaluation metrics (accuracy, confusion matrix)
├── predict.py                # Single-image inference
├── run_ablations.py          # Full ablation suite
├── requirements.txt
└── checkpoints/              # Saved model weights (gitignored)
```

---

## Design Principles

- **Reproducibility first.** Fixed seeds, deterministic data splits, and logged hyperparameters for every reported result.
- **Ablation-driven.** No architectural choice was made without a controlled comparison. See `run_ablations.py` for the full sweep.
- **Edge-deployment ready.** The final model runs inference on CPU in under 10ms per character on a modern laptop, suitable for mobile and embedded deployment.
- **Minimal dependencies.** Pure PyTorch — no heavyweight frameworks, no proprietary tooling.

---

## Potential Applications

- **EdTech:** Automated handwriting assessment for Arabic-language education platforms
- **Document digitization:** OCR pipeline component for Arabic archival and government documents
- **Accessibility tooling:** Real-time character recognition for assistive writing interfaces
- **On-device NLP preprocessing:** Low-latency character segmentation for Arabic NLP pipelines

---

## Keywords

Arabic handwriting recognition · Convolutional neural networks · Squeeze-and-Excitation · CBAM · ResNet · Attention mechanisms · Edge inference · HIJJA dataset · Character recognition · OCR

---

## License

This project is licensed under the [MIT License](LICENSE).
