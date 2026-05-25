# Lightweight Attention-Based CNNs for Arabic Handwriting Recognition (AHR)

This repository contains the official implementation of our study on optimizing Convolutional Neural Networks (CNNs) for Arabic Handwritten Character Recognition. By integrating lightweight attention mechanisms and residual connections, we significantly boost recognition accuracy while maintaining computational efficiency, making the models ideal for low-resource and real-time environments.

The models are trained and evaluated on the HIJJA dataset, a large-scale dataset of isolated Arabic characters written by children.

# Lightweight Attention-Based CNNs for Arabic Handwriting Recognition (AHR)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)
[![Dataset: HIJJA](https://img.shields.io/badge/Dataset-HIJJA-green.svg)](https://github.com/m-aljhani/HIJJA)

An elegant, highly optimized implementation of lightweight Convolutional Neural Networks (CNNs) augmented with structural attention mechanisms and residual connections for **Arabic Handwriting Recognition (AHR)**. 

This repository accompanies our empirical study evaluating resource-efficient architectural modifications on the **HIJJA dataset**—a comprehensive collection of isolated Arabic handwritten characters written by children. By strategically combining a compact **ResNet-13** backbone with **Squeeze-and-Excitation (SE)** blocks, this architecture achieves **93.2% recognition accuracy** while maintaining a highly constrained computational footprint ideal for real-time edge processing and educational technology deployment.

---

## 📖 Table of Contents
- [Executive Abstract](#-executive-abstract)
- [Key Features](#-key-features)
- [Architecture Overview](#-architecture-overview)
- [Experimental Results](#-experimental-results)
- [Repository Structure](#-repository-structure)
- [Installation & Environment Setup](#-installation--environment-setup)
- [Dataset Preparation](#-dataset-preparation)
- [Usage Guidelines](#-usage-guidelines)
- [Keywords](#-keywords)
- [License](#-license)

---

## 📝 Executive Abstract

Arabic Handwriting Recognition (AHR) introduces distinct algorithmic challenges due to the structural complexity, cursiveness, and rich ligatures inherent to Arabic script, exacerbated by the steep variability in individual penmanship. While deep, over-parameterized Convolutional Neural Networks (CNNs) exhibit state-of-the-art performance, deploying them to real-time, low-resource edge environments (such as assistive mobile applications or smart classrooms) requires a meticulous balance between accuracy and structural efficiency.

This work systematically investigates lightweight design adjustments to solve this trade-off. Beginning with a baseline CNN adapted from established literature, we reach **90.2% accuracy** through rigorous hyperparameter tuning. We expand upon this baseline by integrating two prominent channel-wise and spatial attention layers: **Squeeze-and-Excitation (SE)** and the **Convolutional Block Attention Module (CBAM)**. While CBAM provided no statistically significant accuracy lift, the SE block improved classification to **90.7%** with nominal parameter overhead. Transitioning the backbone to a streamlined residual network (**ResNet-13**) pushes performance to **92.8%**. Ultimately, fusing **ResNet-13 with SE blocks** achieves our peak accuracy of **93.2%**, demonstrating that strategic, lightweight enhancements can deliver competitive recognition performance without compromising inference efficiency.

---

## 🚀 Key Features

* ⚡ **Resource-Optimized Layouts:** Explicitly tailored to balance competitive accuracy with microscale parameters, accelerating forward-pass inference on CPUs and low-tier edge platforms.
* 🧠 **Granular Feature Attention:** Complete integration implementations for **Squeeze-and-Excitation (SE)** networks and **Convolutional Block Attention Modules (CBAM)** to systematically evaluate feature recalibration in Arabic orthography.
* 📉 **Stabilized Residual Pipelines:** A customized **ResNet-13** variant designed to preserve spatial feature maps and resolve gradient degradation across fine-grained script variations.
* 🧒 **Robust Child Hand-Composition Evaluation:** Finetuned specifically for the high intra-class variance and irregular strokes standard in children's handwriting profiles present in the HIJJA dataset.

---

## 📐 Architecture Overview

The core paradigm emphasizes modular block integration. Below is the conceptual structural pipeline for the optimal configuration:

```text
Input (Character Image) 
       │
       ▼
 [Conv2D + BatchNorm + ReLU] ──► Initial Representation Layer
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ ResNet-13 Core Block (Iterated)                         │
│   ├── Weight Layer (3x3 Convolution)                   │
│   ├── BatchNorm & ReLU Activation                     │
│   ├── Weight Layer (3x3 Convolution)                   │
│   │                                                    │
│   ▼                                                    │
│ [Squeeze-and-Excitation Block]                         │
│   ├── Global Average Pooling (Spatial Squeeze)         │
│   ├── FC Layer (Reduction) -> FC Layer (Excitation)    │
│   └── Scale Vector Element-wise Multiplication         │
│                                                    │
│   ▼                                                    │
│ ⊕ Residual Skip Connection Addition                    │
└────────────────────────────────────────────────────────┘
       │
       ▼
 [Global Average Pooling]
       │
       ▼
 [Dense Softmax Classifier] ──► Class Prediction (28 Arabic Alphabetical Profiles)
