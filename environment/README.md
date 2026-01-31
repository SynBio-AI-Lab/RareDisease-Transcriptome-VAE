# Environment Setup

This directory contains the configuration files required to set up the computational environment for this project. 

Following these steps ensures full reproducibility of the VAE training, temperature scaling experiments, and biomarker identification results.

## Prerequisites

- **Python Version**: Python 3.9 or higher is recommended for compatibility with PyTorch 2.0+ and the latest Scikit-learn features.
- **Hardware**: While the VAE can be trained on a CPU due to the small sample size ($n < 100$), an NVIDIA GPU with CUDA support is recommended for faster execution of the 5-run robustness checks.

## Installation
We recommend using a virtual environment to avoid conflicts with your system-wide Python packages.

### 1. Create a Virtual Environment
You may use either `venv` or Conda.

**Using venv:**
```bash
python -m venv venv
```

**Using Conda**
```bash
conda create -n raredisease-vae python=3.9
conda activate raredisease-vae
```

### 2. Activate the Environment

If you used **venv**:

- **Windows**:

```bash
venv\Scripts\activate
```

- **macOS / Linux**:

```bash
source venv/bin/activate
```

If you used Conda, the environment is already activated after:
```bash
conda activate raredisease-vae
```



### 3. Install Dependencies
Install the required packages:

```bash
pip install --upgrade pip
pip install -r environment/requirements.txt
```

## Key Dependencies Overview
As specified in `requirements.txt`, this project relies on the following core stack:

- **PyTorch (>=2.0)**: Main deep learning framework for the VAE architecture.

- **Scikit-learn (>=1.2)**: PCA, evaluation metrics, and data scaling.

- **Pandas & NumPy**: Core data manipulation for transcriptome matrices.

## Verification
To verify the setup:
```bash
python - <<EOF
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
EOF
```
