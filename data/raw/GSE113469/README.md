# GSE113469: External Validation Dataset

This directory is reserved for publicly available transcriptome data from NCBI GEO used for external validation in this project.

Raw files are not included in this repository and can be downloaded directly from NCBI.

## Dataset Information

- **Accession**: GSE113469
- **Organism**: Homo sapiens
- **Samples**: 37 small intestinal mucosa samples

## Download Instructions

1. Visit the NCBI GEO page:  
   https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE113469
2. Download the **Series Matrix File(s)**.
3. Place the downloaded files into this directory:  
   `data/raw/GSE113469/`

## Expected Files (example)

After download, this directory may contain:

- `GSE113469_series_matrix.txt.gz`

The validation pipeline will automatically locate this file during evaluation.
