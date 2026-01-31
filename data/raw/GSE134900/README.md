# GSE134900: Primary Training Dataset

This directory is reserved for publicly available transcriptome data from NCBI GEO used as the primary training dataset for this project.

Due to data redistribution policies, raw files are not included in this repository and can be downloaded directly from NCBI.

## Dataset Information

- **Accession**: GSE134900
- **Organism**: Homo sapiens
- **Samples**: 96 duodenal biopsy samples (Celiac Disease vs Control)
- **Platform**: GPL16791 (Illumina HiSeq 2500)

## Download Instructions

1. Visit the NCBI GEO page:  
   https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE134900
2. Download the **Series Matrix File(s)** or the **RAW supplementary files**.
3. Place the downloaded files into this directory:  
   `data/raw/GSE134900/`

## Expected Files (example)

After download, this directory may contain files such as:

- `GSE134900_series_matrix.txt.gz`

The preprocessing script will automatically detect these files and generate processed matrices in `data/processed/`.
