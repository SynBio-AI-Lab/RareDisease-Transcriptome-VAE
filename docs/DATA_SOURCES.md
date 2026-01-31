# Data Sources and Preprocessing

This document describes the public data sources used in this project and the preprocessing steps applied before model training and evaluation.

All datasets used in this study are publicly available from the NCBI Gene Expression Omnibus (GEO) and are not redistributed in this repository in accordance with GEO data usage policies.

## Primary Training Dataset

### Accession: GSE134900

- Source: NCBI GEO
- Organism: Homo sapiens
- Tissue: Duodenal biopsy
- Samples: 96 (Celiac Disease vs Control)
- Platform: GPL16791 (Illumina HiSeq 2500)  
URL: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE134900

This dataset is used as the main training dataset for the VAE model and as the basis for synthetic data generation.

## External Validation Dataset

### Accession: GSE113469

- Source: NCBI GEO
- Organism: Homo sapiens
- Tissue: Small intestinal mucosa
- Samples: 37  
URL: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE113469

This dataset is used to evaluate model generalization on an independent cohort.

## Data Download Policy

Raw data files are not included in this repository.

To reproduce the experiments:

1. Download the **Series Matrix File(s)** from the GEO pages above.
2. Place the files into:
   - `data/raw/GSE134900/`
   - `data/raw/GSE113469/`

The preprocessing pipeline will automatically detect and process these files.

## Preprocessing Pipeline

The following steps are automatically performed by the preprocessing script:

1. Extraction of expression matrices from GEO Series Matrix files
2. Gene ID alignment and filtering
3. Log transformation
4. Standard scaling
5. Export of processed matrices to `data/processed/`

Script location:

```text
src/datasets/download_and_preprocess.py
```

## Synthetic Data Generation

Synthetic datasets are generated only from the processed version of GSE134900 using the trained VAE model.

These synthetic datasets are provided via Zenodo due to size limitations and can be placed into:

```text
data/synthetic/
```

## Compliance Statement
This repository does not redistribute raw GEO data.

All users must download the datasets directly from NCBI GEO under their respective terms of use.

## References
- NCBI Gene Expression Omnibus (GEO)
https://www.ncbi.nlm.nih.gov/geo/
- GSE134900 and GSE113469 dataset pages (linked above)

