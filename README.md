# Lightweight Attention-Based CNNs for Arabic Handwriting Recognition

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Framework: TensorFlow](https://img.shields.io/badge/Framework-TensorFlow-ff6f00.svg)](https://www.tensorflow.org/)
[![Dataset: HIJJA](https://img.shields.io/badge/Dataset-HIJJA-green.svg)](https://github.com/m-aljhani/HIJJA)

An empirical study evaluating lightweight CNN modifications for Arabic Handwriting Recognition (AHR) on the **HIJJA dataset** — a large-scale collection of isolated Arabic characters written by children. By combining a compact **ResNet-13** backbone with **Squeeze-and-Excitation (SE)** attention, the final architecture achieves **93.2% test accuracy** while remaining viable for real-time edge deployment.

---

## Table of Contents

- [Abstract](#abstract)
- [Results](#results)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Dataset](#dataset)

---

## Abstract

Arabic Handwriting Recognition poses distinct challenges: cursive connectivity, rich ligatures, and high intra-class variance from irregular child penmanship. Over-parameterised CNNs achieve strong accuracy but are impractical for low-resource environments such as assistive mobile apps or smart classrooms.

This study systematically investigates lightweight design adjustments:

1. A **baseline CNN** replicating the original HIJJA paper reaches **90.2% accuracy** after hyperparameter tuning.
2. Adding **CBAM** (channel + spatial attention) provides no statistically significant lift.
3. Adding an **SE block** (channel recalibration only) improves accuracy to **90.7%** with negligible parameter overhead.
4. Replacing the backbone with **ResNet-13** pushes performance to **92.8%**.
5. Fusing **ResNet-13 + SE** achieves the peak of **93.2%**, confirming that strategic lightweight enhancements can close the gap with larger models.

---

## Results

| Model            | Test Accuracy | Δ vs Baseline | Attention          | Backbone    |
|------------------|:-------------:|:-------------:|--------------------|-------------|
| Baseline CNN     | 90.2%         | —             | None               | Vanilla CNN |
| Baseline + CBAM  | ~90.2%        | ≈ 0           | CBAM (ch + spatial)| Vanilla CNN |
| Baseline + SE    | 90.7%         | +0.5pp        | SE (channel only)  | Vanilla CNN |
| ResNet-13        | 92.8%         | +2.6pp        | None               | ResNet-13   |
| **ResNet-13 + SE** | **93.2%** | **+3.0pp**    | SE (channel only)  | ResNet-13   |

Key findings:

- Residual connections are the largest single driver of improvement.
- SE attention provides a consistent marginal gain (~0.4pp) at negligible cost.
- CBAM's spatial branch is redundant for small (32×32) isolated character images.

---

## Architecture

The best-performing configuration (ResNet-13 + SE):

```
Input (32 × 32 × 1 grayscale)
        │
        ▼
 Conv2D 3×3 · BN · ReLU        ← stem
        │
        ▼
┌──────────────────────────────────┐
│  Residual Block (×3 stages)      │  Filters: 64 → 128 → 256
│    Conv2D 3×3 · BN · ReLU       │
│    Conv2D 3×3 · BN              │
│         │                       │
│   ┌─────┴──────┐                │
│   │  SE Block  │                │  Global avg pool → FC↓ → ReLU
│   │            │                │  → FC↑ → Sigmoid → channel scale
│   └─────┬──────┘                │
│         ⊕ skip connection       │
└──────────────────────────────────┘
        │
        ▼
 Global Average Pooling
        │
        ▼
 Dense · Softmax → 28 classes
```

**Training configuration:** Adam · lr 0.001 · batch 64 · EarlyStopping (patience 15) · ReduceLROnPlateau (patience 7) · seed 42.

---

## Repository Structure

```
.
├── baseline.py      # Baseline CNN replicating the original HIJJA paper
├── SE_Net.py        # SE attention placement and reduction ratio sweep
├── CBAM.py          # CBAM attention (R × K) grid search
├── ResNet.py        # ResNet block type, depth, and SE fusion experiments
├── load_data.py     # Shared data loading and preprocessing utilities
├── index.html       # Interactive results visualisation (open in browser)
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

You also need a Kaggle API key configured for dataset download via `kagglehub`. See [Kaggle API docs](https://www.kaggle.com/docs/api).

---

## Usage

Each script is self-contained and downloads the HIJJA dataset automatically on first run.

```bash
# Baseline replication
python baseline.py

# SE attention sweep
python SE_Net.py

# CBAM grid search
python CBAM.py

# ResNet + SE experiments
python ResNet.py
```

Results are printed as a summary table at the end of each run.

To view the interactive results visualisation, open `index.html` in any browser — no server required.

---

## Dataset

**HIJJA** — Handwritten Arabic Characters Dataset  
28 isolated Arabic character classes, written by school-age children.  
Images: 32×32 grayscale, provided as CSV (pixel rows).

- Paper: *Hijja: A Handwritten Arabic Characters Dataset* (Altwaijry & Al-Turaiki)
- Kaggle: `mohamedgreshamahdi/hijja-dataset`
- GitHub: https://github.com/m-aljhani/HIJJA
