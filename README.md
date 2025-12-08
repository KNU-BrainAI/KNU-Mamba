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

2. Install PyTorch
Install the version of PyTorch compatible with your CUDA version (e.g., CUDA 11.8):

Bash

pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
3. Install Mamba Dependencies
Mamba relies on optimized CUDA kernels (causal-conv1d and mamba-ssm):

Bash

pip install causal-conv1d>=1.2.0
pip install mamba-ssm
4. Install KNU-Mamba
Clone this repository and install the remaining requirements:

Bash

git clone [https://github.com/KNU-BrainAI/KNU-Mamba.git](https://github.com/KNU-BrainAI/KNU-Mamba.git)
cd KNU-Mamba
pip install -r requirements.txt
pip install -e .
💻 Usage
Basic Inference
Below is a minimal example of initializing the model and running a forward pass:

Python

import torch
from knu_mamba import MambaModel

# Check for GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Configuration parameters
batch_size = 4
seq_len = 2048
d_model = 128
d_state = 16  # State expansion factor

# Initialize the model
model = MambaModel(
    d_model=d_model,
    d_state=d_state,
    d_conv=4,
    expand=2
).to(device)

# Create dummy input (Batch, Seq_Len, Dimension)
x = torch.randn(batch_size, seq_len, d_model).to(device)

# Forward pass
output = model(x)

print(f"Input Shape: {x.shape}")
print(f"Output Shape: {output.shape}")
Training
To run a training experiment (e.g., on a sample dataset), run:

Bash

python train.py --config configs/experiment_base.yaml
📂 Project Structure
Plaintext

KNU-Mamba/
├── configs/            # YAML configuration files for experiments
├── knu_mamba/          # Main package source code
│   ├── modules/        # Mamba blocks, SSM layers, and Normalization
│   ├── ops/            # Custom CUDA kernels and low-level ops
│   ├── models/         # Full model architectures (e.g., MambaLM, VisionMamba)
│   └── utils/          # Data loaders and logging utilities
├── examples/           # Jupyter notebooks and demo scripts
├── tests/              # Unit tests for correctness
├── train.py            # Main training entry point
├── requirements.txt    # Python dependencies
└── README.md           # Documentation
🔬 Research Context
This repository supports the Brain AI Lab's mission to bridge neuroscience and machine learning. Specifically, we investigate Mamba for:

Brain-Inspired Computing: Modeling neural dynamics and spiking neural networks (SNNs).

Bio-Signal Analysis: Efficient processing of high-frequency time-series data (e.g., EEG, fMRI) for Brain-Computer Interfaces (BCI).

Visual Perception: Long-range dependency modeling in high-resolution visual tasks.

📜 Citation
If you use this codebase in your research, please cite the original Mamba paper:

Code snippet

@article{gu2023mamba,
  title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author={Gu, Albert and Dao, Tri},
  journal={arXiv preprint arXiv:2312.00752},
  year={2023}
}
If you refer to this specific implementation or KNU Brain AI Lab resources:

Code snippet

@misc{knumamba2024,
  author = {KNU Brain AI Lab},
  title = {KNU-Mamba: Research Implementation of Mamba SSM},
  year = {2024},
  publisher = {GitHub},
  howpublished = {\url{[https://github.com/KNU-BrainAI/KNU-Mamba](https://github.com/KNU-BrainAI/KNU-Mamba)}}
}
👥 Contributors
Brain AI Lab, Kyungpook National University

Md Tanvir Islam (Lead Maintainer)

Prof. Sangtae Ahn (Advisor)

📄 License
This project is licensed under the MIT License. See the LICENSE file for details.
