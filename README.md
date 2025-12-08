# KNU-Mamba

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![KNU Brain AI](https://img.shields.io/badge/Lab-KNU%20Brain%20AI-red)](https://knu-brainai.github.io/)

**KNU-Mamba** is a PyTorch-based implementation of the Mamba State Space Model (SSM) architecture, developed and maintained by the **Kyungpook National University (KNU)**.

This repository serves as a codebase for exploring efficient sequence modeling using Mamba, capable of handling long-range dependencies with linear computational complexity.

> **Note**: This project builds upon the original Mamba architecture proposed by Gu & Dao (2023).

## 🚀 Features

- **Efficient Implementation**: optimized for fast training and inference on GPUs.
- **Modular Design**: Easy integration into existing deep learning pipelines (e.g., for Computer Vision, NLP, or Time-Series analysis).
- **Research Ready**: Structured for experimentation with different state-space configurations.

## 🛠️ Installation

### Prerequisites
- Linux
- NVIDIA GPU with CUDA 11.6+
- Python 3.8+
- PyTorch 1.12+

### Set up the Environment
We recommend using Conda to manage the environment:

```bash
conda create -n knu-mamba python=3.10
conda activate knu-mamba
