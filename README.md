# Robust Generative Augmentation of Rare Disease Transcriptomes: Sampling Temperature Scaling to Mitigate Mode Collapse in VAEs

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18431155.svg)](https://doi.org/10.5281/zenodo.18431155) 
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This repository provides the official implementation of a VAE-based generative data augmentation pipeline designed to address the **"small $n$, large $p$"** challenge in rare disease transcriptomics. By introducing **Adaptive Sampling Temperature Scaling**, this framework effectively mitigates mode collapse in data-scarce environments and ensures biological diversity in synthetic samples.

## Research Highlights

- **Adaptive Temperature Scaling**: We empirically demonstrate that temperature values in the range of 2.0–3.0 prevent mode collapse in ultra-small cohorts ($n < 100$), enabling more effective exploration of the latent biological manifold.

- **High Statistical Fidelity**: The generated synthetic data achieves exceptional similarity to real patient data, with **gene-wise $R^2$ values consistently above 0.99 (mean = 0.9988)**.

- **Clinical Utility (TSTR)**: Models trained on synthetic data (Train-on-Synthetic, Test-on-Real) achieved an accuracy of $99.3\% \pm 1.5\%$ and an AUC of $0.990 \pm 0.023$, demonstrating strong downstream utility.

- **Biological Validity**: Feature importance analysis identified pathologically significant biomarkers, including TRPM6, APOA1, and TREH, which are consistent with the intestinal absorption mechanisms detailed in *Sleisenger and Fordtran's Gastrointestinal and Liver Disease (12th Ed.)*.


## Repository Structure
```text
RareDisease-Transcriptome-VAE/
├── environment/         # Environment setup (requirements.txt)
├── data/
│   ├── raw/             # Guidelines for downloading NCBI GEO datasets
│   ├── processed/       # Log-transformed and scaled matrices
│   └── synthetic/       # Generated 10-fold augmented synthetic data (.csv)
├── models/              # Trained VAE and validation model checkpoints (.pt)
├── src/                 # Source code for analysis, generation, and visualization
├── results/             # Statistical reports and 5-run robustness logs
├── supplementary/       # Comparative analysis with Vanilla GAN baselines
└── docs/                # Citation guidelines and data source documentation
```

## Getting Started

### 1. Installation
```bash
git clone https://github.com/SynBio-AI-Lab/RareDisease-Transcriptome-VAE.git
cd RareDisease-Transcriptome-VAE
pip install -r environment/requirements.txt
```
### 2. Required Artifacts (Models & Data)

Large artifacts are not included in this GitHub repository and are hosted on Zenodo.

**Zenodo DOI**: https://doi.org/10.5281/zenodo.18431155

Download the Zenodo archive and extract it at the root of this repository.

The archive mirrors the directory structure of this project and will automatically
place files (models, synthetic data, processed matrices, and results) into their
correct locations.

## Reproducibility
To reproduce the 5-run independent robustness check reported in the paper (Table A1), run the following script:
```bash
python src/robustness_check.py
```
This script assesses the model's stability against initialization noise, ensuring a minimal performance variance ($\sigma \approx 0.015$).

## Citation
If you find this work useful for your research, please cite it as follows:
```bibtex
@article{Song2026VAE,
  title={Robust Generative Augmentation of Rare Disease Transcriptomes: Sampling Temperature Scaling to Mitigate Mode Collapse in VAEs},
  author={Song, Min Kyung},
  journal={Bioinformatics (Submitted)},
  year={2026},
  doi={10.5281/zenodo.18431155},
  url={https://github.com/SynBio-AI-Lab/RareDisease-Transcriptome-VAE}
}
```


**Contact**: Min Kyung Song (songseoul5440@gmail.com) 

**Affiliation**: Department of Convergence Technology Management, Kwangwoon University
